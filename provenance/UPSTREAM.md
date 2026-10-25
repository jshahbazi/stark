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

Active Rust crates use the `stark-engine`, `stark-engine-macros`, `stark-score` and `stark-generate` names. The inherited crates under `crates/legacy` keep their Uzu-era names and are not part of the product. These upstream identifiers are kept on purpose:

- Uzu Git history, upstream license, revision metadata and original patch hashes.
- Mirai model registry and provider IDs, API schema names, model downloads and dependency repository URLs inside the inherited crates.
- Registry backend IDs `uzu`/`Uzu`, the serialized telemetry field `uzu_version`, and the fallback key `__uzu_unparsed_arguments`, which preserve external compatibility of the inherited bindings.
- The historical `UZU_CAPTURE_` environment names, which both workers clear together with `STARK_CAPTURE_` names.
- Mirai attribution and artwork in the inherited sample application.

The root `Package.swift` describes the inherited Uzu Swift binding and expects a locally built `stark.xcframework` that no workflow produces; the menu bar app is built with `swiftc` directly. Inherited self-hosted CI, benchmark deployments, package publishing and cloud artifact destinations are disabled.
