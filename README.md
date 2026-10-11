# Stark

Stark is an experimental Rust and Metal inference engine for Apple silicon, forked from [Uzu](https://github.com/trymirai/uzu). It adds exact-token probability scoring and optimizations developed for JevK5.

The fork starts at Uzu 0.6.1, commit `8d01581474760bd1fc7f4a547ff1d02f7dada437`, and retains its Git history and MIT license.

## Build the scoring engine

Use an Apple silicon Mac with Xcode's Metal tools available. Rust 1.98.1 is pinned. Model files must be converted to the Uzu model format; model weights are not included. The [conversion prototype](tools/converter/README.md) provides streaming dense Qwen3 conversion and a pinned Lalamo fallback for broader conversion coverage.

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

[Validation](docs/VALIDATION.md) records the BF16 probability and calibration checks for the Stark build. Precise performance testing remains deferred.

## Scope and attribution

The supported path for this experimental fork is the native Rust/Metal scoring worker. Inherited Python, TypeScript, Swift, CLI, and playground sources have been renamed, but their binary distributions and general generation behavior have not been qualified for Stark. No Stark packages or Swift XCFramework are published. Inherited cloud release and benchmark workflows are disabled.

See [rename compatibility](docs/RENAMING.md), [upstream provenance](provenance/UPSTREAM.md), and [LICENSE](LICENSE). Uzu and Mirai attribution and external provider identities remain where required by licensing, model compatibility, or historical evidence.

## Security and contribution checks

Report vulnerabilities through [private reporting](SECURITY.md). [CI and security checks](docs/CI.md) cover synthetic scoring tests, process ownership, dependency advisories, and static analysis. Model-dependent validation remains private and separate from public CI.
