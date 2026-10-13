# Stark

Stark is an experimental Rust and Metal inference engine for Apple silicon, forked from [Uzu](https://github.com/trymirai/uzu). It adds exact-token probability scoring and optimizations developed for JevK5.

The fork starts at Uzu 0.6.1, commit `8d01581474760bd1fc7f4a547ff1d02f7dada437`, and retains its Git history and MIT license.

## Build the scoring engine

Use an Apple silicon Mac with Xcode's Metal tools available. Rust 1.98.1 is pinned. Model files must be converted to the Uzu model format; model weights are not included. The [model converter](tools/converter/README.md) provides independent streaming conversion for selected dense Qwen, Llama, Mistral, Gemma text, Phi-4-mini, Granite, SmolLM, LFM2 and LLaMBA profiles, including Qwen3.5 text/DeltaNet, plus affine MLX 4/8-bit Qwen3, Qwen3.5 text, Llama, SmolLM3, Gemma text, LFM2 and LLaMBA checkpoints. Exact checkpoint qualification and runtime status are recorded separately; unsupported combinations fail explicitly.

ModernBERT sequence classifiers use the converter's explicit `--model-kind classifier` path. Classifier conversion, native loading and classification execution have separate evidence and status.

```sh
cargo build --release --locked -p stark-score --features optimized
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
./target/release/stark-score --score-model-path /absolute/path/to/model
```

The worker uses a private JSON-lines stdin/stdout protocol. Request and response content belongs in private pipes or files. It suppresses native diagnostic output and capture variables. [Scoring protocol](docs/SCORING.md) describes probabilities, calibration, and optional consecutive caching.

## Model conversion and Lalamo comparison

The [Qwen3 GGUF experiment](docs/QWEN3_GGUF_IMPORT_EXPERIMENT.md) imports official Qwen3-0.6B Q8_0 weights into the existing native compressed format. The [prefill investigation](docs/QWEN3_GGUF_PREFILL_OPTIMIZATION.md) improves the imported model's prefill throughput by about 15% at 512 and 1,024 tokens on one M5 Max, with unchanged compressed weights and exact outputs on the tested cases. General GGUF support and generation quality remain unqualified.

The [milestone 6 classifier benchmarks](docs/MODEL_CONVERSION_MILESTONE_6.md) measured public real-weight ModernBERT fixtures with explicitly recorded synthetic chat templates. BF16-source conversion took **0.562 s versus Lalamo's 1.137 s**; F32-source conversion took **0.861 s versus 1.312 s**, with overlapping F32 ranges. Median times were **34–51% lower**, with **85–90% less sampled process-tree RSS**. All twenty outputs pass full parity; private native checks prove loading only. These fixtures do not qualify the inaccessible catalog routers or establish classification quality.

`stark-convert` implements 85 of the 110 pinned Lalamo built-ins plus four Stark extensions, including 40 MLX profiles and two ModernBERT classifier profiles. Its runtime dependencies exclude Lalamo, JAX and PyTorch. Conversion uses an 8 MiB transformation workspace and preserves exact tensor/spec/config/tokenizer parity for qualified tuples. Full initial and final source verification remain enabled. Twenty-seven exact BF16 tuples have named-build load-only qualifications. [Milestone 6](docs/MODEL_CONVERSION_MILESTONE_6.md) adds classifier output configurations, tokenizer/label validation and future-checkpoint conversion with `--model-kind classifier`. Both pinned classifier repositories are inaccessible to the current account and remain unqualified; public classifier validation fixtures have separate evidence. [Milestone 5](docs/MODEL_CONVERSION_MILESTONE_5.md) retains its LLaMBA/Mamba2 qualifications.

Conversion adapters are tracked by architecture and weight format. Checkpoint metadata, source revisions and exact qualifications are tracked separately. The new explicit [`hf-generic` path](tools/converter/README.md#future-checkpoints-and-architecture-tracking) lets a future Hugging Face checkpoint reuse a supported adapter with its own self-contained metadata, without adding a catalog entry. Its complete config/layout still must pass validation, and it starts native-unqualified.

The [milestone 5 LLaMBA benchmarks](docs/MODEL_CONVERSION_MILESTONE_5.md) measured 4-bit MLX at **2.142 s versus Lalamo's 2.225 s**, with overlapping ranges, and dense F32-source conversion at **5.791 s versus 2.105 s**. Dense conversion is 2.75 times as slow; its latency objective remains open. Both use **96–97% less sampled process-tree RSS** and pass exact parity and named-build loading. All twenty observations are retained; candidate timing includes stronger full source verification.

The preserved [milestone 4 MLX benchmarks](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md) measured **28–71% lower median complete conversion time than Lalamo**, with **92–95% less sampled process-tree RSS**, for Qwen3.5-0.8B text and LFM2-350M in both 4-bit and 8-bit formats. All forty timed outputs pass exact parity; the twenty candidate runs match their exact named-build load-only qualifications. These are five warm-file observations per backend/tuple on one host; inference speed and quality remain unverified.

The preserved [milestone 4 dense benchmarks](docs/MODEL_CONVERSION_MILESTONE_4.md) measured LFM2-350M at **1.233 s versus Lalamo's 1.688 s** and Qwen3.5-0.8B text at **3.525 s versus 2.707 s**. LFM2's median is 27% lower; Qwen3.5's is 30% higher. Median sampled process-tree RSS is 90–91% lower for both. Five-repeat ranges are broad and overlap; all observations and the failed Qwen3.5 latency objective remain recorded. Both outputs pass exact parity and named-build native loading, with zero scoring calls.

The preserved milestone 3 sequential coefficient export reduced median complete conversion time by **16–38%** against its previous branch implementation across six MLX tuples. Five measured medians are lower than pinned Lalamo, with **90.7–96.6% less sampled process-tree RSS**. All outputs match the existing exact named-build BF16 load-only qualifications.

| MLX checkpoint, BF16 output | Stark median | Lalamo median | Lower sampled RSS |
| --- | ---: | ---: | ---: |
| Llama3.2-1B 4-bit | 1.256 s | 2.397 s | 94.2% |
| Llama3.2-1B 8-bit | 2.202 s | 2.604 s | 96.6% |
| SmolLM3-3B 4-bit | 2.779 s | 3.377 s | 95.4% |
| SmolLM3-3B 8-bit | 5.083 s | 4.233 s | 96.2% |
| Gemma3-1B 4-bit | 1.551 s | 2.580 s | 90.7% |
| Gemma3-1B 8-bit | 2.464 s | 2.537 s | 90.8% |

These are five-repeat, warm-cache complete-command measurements on one Apple silicon host. Stark retains full initial/final source verification and stronger receipt work. SmolLM3 8-bit remains **20% slower** than Lalamo. Gemma 8-bit's median advantage is about 3%; its observed timing ranges overlap, as do Llama 8-bit's. Broad latency parity remains open. Conversion measurements grant no inference speed or quality qualification. See [current evidence and protocol](docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md), [independent review](docs/MODEL_CONVERSION_MILESTONE_3_LATENCY_REVIEW.md), [preserved dense results](docs/MODEL_CONVERSION_MILESTONE_3_DENSE.md), and [converter usage](tools/converter/README.md).

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
