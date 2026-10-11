# Stark



## Build the scoring engine

Use an Apple silicon Mac with Xcode's Metal tools available. Rust 1.98.1 is pinned. Model files must already be converted to the Uzu model format; model weights are not included.

```sh
cargo build --release --locked -p stark-score --features optimized
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
./target/release/stark-score --score-model-path /absolute/path/to/model
```

The worker uses a private JSON-lines stdin/stdout protocol. Request and response content belongs in private pipes or files. It suppresses native diagnostic output and capture variables. [Scoring protocol](docs/SCORING.md) describes probabilities, calibration, and optional consecutive caching.

## Changes from Uzu

- Raw next-token logits, full-vocabulary log probabilities, and temperature-calibrated probabilities over requested candidate tokens.
- Direct scoring topology, deferred state initialization, fewer Metal submissions, and pruning of unobserved final-layer work.
- Direct and compact delta convolution, reusable RoPE values, and KV-only attention preparation for scoring.
- An optional one-entry cache of the completed vocabulary distribution for identical consecutive token sequences and context size.

The `optimized` feature selects the six accepted optional engine optimizations. Consecutive caching requires the separate `--consecutive-cache` runtime flag and defaults off. These changes do not establish faster general text generation or cross-model quality acceptance.


## Scope and attribution

The supported path for this experimental fork is the native Rust/Metal scoring worker. Inherited Python, TypeScript, Swift, CLI, and playground sources have been renamed, but their binary distributions and general generation behavior have not been qualified for Stark. No Stark packages or Swift XCFramework are published. Inherited cloud release and benchmark workflows are disabled.

See [rename compatibility](docs/RENAMING.md), [upstream provenance](provenance/UPSTREAM.md), and [LICENSE](LICENSE). Uzu and Mirai attribution and external provider identities remain where required by licensing, model compatibility, or historical evidence.
