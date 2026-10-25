# Upstream and import provenance

- Upstream: https://github.com/trymirai/uzu
- Version: 0.6.1
- Base commit: `8d01581474760bd1fc7f4a547ff1d02f7dada437`
- License: MIT, original Mirai Tech Inc. copyright retained in the root LICENSE.
- Import commit: `40674ab0ef5bd3e664c41da59d3feb9a6df90960`, nine accepted scoring and optimization patches applied in their recorded order.

The import contains the scoring API, topology and state changes, Rust/Metal optimizations, RoPE reuse, direct delta-convolution, and attention preparation optimizations. The standalone scoring crate additionally contains the default-off consecutive distribution cache and full-vocabulary digest probe. Renaming optimization features preserves their implementation and arithmetic.

The pinned upstream ancestry, root license, original upstream README, dependency lock, and JevK5 fixture licenses are preserved. Earlier fork commits were rewritten to keep only standalone engine source and neutral optimization names. Source-bound receipts from the previous history are replaced by fresh build and correctness validation records for the rewritten source.

Model weights, private fixtures, runtime executables, and host application archives are excluded. The original pinned upstream README is stored verbatim as [UPSTREAM_README.md](UPSTREAM_README.md); its relative links refer to the upstream tree and are not expected to resolve here. The current build and validation receipts identify the complete standalone source and executable hashes.

## Naming and compatibility

The Rust workspace contains only the `stark-engine`, `stark-engine-macros`, `stark-score` and `stark-generate` crates. These upstream identifiers are kept on purpose:

- Uzu Git history, upstream license, revision metadata and original patch hashes.
- The historical `UZU_CAPTURE_` environment names, which both workers clear together with `STARK_CAPTURE_` names.

The menu bar app is built with `swiftc` directly. Inherited self-hosted CI, benchmark deployments, package publishing and cloud artifact destinations are disabled.

## Removed inherited code

On 2026-10-25 the inherited Uzu code that no product component used was removed, together with the dependencies only it needed:

- The seventeen crates under `crates/legacy` (the `stark` engine facade and its Python, Swift and TypeScript bindings, the HTTP server, `cli`, `cli-tools`, the model registry and download manager, and their support crates), including their example and binding lockfiles.
- `vendor/napi-rs`, the patched napi crates used only by the TypeScript binding, and the matching `[patch.crates-io]` entries.
- The root `Package.swift`, which described the Swift binding, and `apps/playground`, the sample application built on it.
- `.config/platforms.toml`, read only by `cli-tools`.

`stark-engine` used four plain types from the inherited `shoji` crate; they are now defined in the engine with the same fields. The engine unit suite used the inherited engine to download its test model and the inherited `keisoku` crate for a benchmark cooldown; it now reads a converted model from `STARK_TEST_MODEL_PATH` and has no cooldown. `Cargo.lock` went from 813 to 263 packages with no version changes. The removed files remain in Git history at the commit before the removal, and the originals are at the upstream base commit above. `provenance/BUILD_RECEIPT.json` still lists hashes of the removed files because it records the source of the build it describes.
