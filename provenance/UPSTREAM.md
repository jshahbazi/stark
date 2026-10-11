# Upstream and import provenance

- Upstream: https://github.com/trymirai/uzu
- Version: 0.6.1
- Base commit: `8d01581474760bd1fc7f4a547ff1d02f7dada437`
- License: MIT, original Mirai Tech Inc. copyright retained in the root LICENSE.
- Import commit: `40674ab0ef5bd3e664c41da59d3feb9a6df90960`, nine accepted scoring and optimization patches applied in their recorded order.

The import contains the scoring API, topology and state changes, Rust/Metal optimizations, RoPE reuse, direct delta-convolution, and attention preparation optimizations. The standalone scoring crate additionally contains the default-off consecutive distribution cache and full-vocabulary digest probe. Renaming optimization features preserves their implementation and arithmetic.

The pinned upstream ancestry, root license, original upstream README, dependency lock, and JevK5 fixture licenses are preserved. Earlier fork commits were rewritten to keep only standalone engine source and neutral optimization names. Source-bound receipts from the previous history are replaced by fresh build and correctness validation records for the rewritten source.

Model weights, private fixtures, runtime executables, and host application archives are excluded. The original pinned upstream README is stored alongside this document. The current build and validation receipts identify the complete standalone source and executable hashes.
