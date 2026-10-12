# Stark

Stark is an experimental Rust and Metal inference engine for Apple silicon, forked from [Uzu](https://github.com/trymirai/uzu). It adds exact-token probability scoring and optimizations developed for JevK5.

The fork starts at Uzu 0.6.1, commit `8d01581474760bd1fc7f4a547ff1d02f7dada437`, and retains its Git history and MIT license.

## Build the scoring engine

Use an Apple silicon Mac with Xcode's Metal tools available. Rust 1.98.1 is pinned. Model files must be converted to the Uzu model format; model weights are not included. The [model converter](tools/converter/README.md) provides independent streaming conversion for selected dense Qwen, Llama, Mistral, Gemma text, Phi-4-mini, Granite and SmolLM profiles, plus affine MLX 4/8-bit Qwen3, Llama, SmolLM3 and Gemma text checkpoints. Exact checkpoint qualification and runtime status are recorded separately; unsupported combinations fail explicitly.

```sh
cargo build --release --locked -p stark-score --features optimized
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
./target/release/stark-score --score-model-path /absolute/path/to/model
```

The worker uses a private JSON-lines stdin/stdout protocol. Request and response content belongs in private pipes or files. It suppresses native diagnostic output and capture variables. [Scoring protocol](docs/SCORING.md) describes probabilities, calibration, and optional consecutive caching.

## Model conversion and Lalamo comparison

`stark-convert` implements 54 of the 110 pinned Lalamo built-ins plus four Stark extensions, including 21 MLX profiles. Its runtime dependencies exclude Lalamo, JAX and PyTorch. MLX conversion streams existing packed weights and transforms affine coefficients within an 8 MiB workspace, preserving exact tensor/spec/config/tokenizer parity for qualified tuples. Full initial and final source verification remain enabled.

The six newly measured MLX tuples used **90.7–96.6% less sampled process-tree RSS** than pinned Lalamo. Llama3.2-1B 4-bit and Gemma3-1B 4-bit conversion were **1.24× and 1.23× faster**, respectively. All six passed exact parity and named-build BF16 load-only checks.

| MLX checkpoint, BF16 output | Stark median | Lalamo median | Lower sampled RSS |
| --- | ---: | ---: | ---: |
| Llama3.2-1B 4-bit | 1.845 s | 2.292 s | 94.1% |
| Llama3.2-1B 8-bit | 2.233 s | 2.137 s | 96.6% |
| SmolLM3-3B 4-bit | 3.979 s | 2.809 s | 95.4% |
| SmolLM3-3B 8-bit | 6.019 s | 3.544 s | 96.2% |
| Gemma3-1B 4-bit | 1.991 s | 2.444 s | 90.7% |
| Gemma3-1B 8-bit | 2.655 s | 2.171 s | 92.5% |

These are three-repeat, warm-cache complete-command measurements on one Apple silicon host. Stark performs stronger source verification and receipt work. Four of the six tuples have slower median conversion times, so broad latency parity remains open. The earlier Qwen3-0.6B MLX packet measured 1.78×/1.21× speedups at its separately recorded source state. Conversion results do not qualify inference speed or quality. See [current evidence and protocol](docs/MODEL_CONVERSION_MILESTONE_3_DENSE.md), [earlier Qwen3 results](docs/MODEL_CONVERSION_MILESTONE_3.md), and [converter usage](tools/converter/README.md).

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
