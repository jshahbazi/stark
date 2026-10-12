# Stark

**Fast model preparation and exact probability scoring on Apple silicon.**

Stark combines a Rust and Metal inference engine with an independent streaming converter for Hugging Face models. Prepare dense and existing MLX 4/8-bit checkpoints locally, then use direct probability scoring for decision models such as JevK5.

Stark is an experimental fork of [Uzu](https://github.com/trymirai/uzu), starting at Uzu 0.6.1, commit `8d01581474760bd1fc7f4a547ff1d02f7dada437`. Upstream Git history, copyright and MIT licensing are preserved.

## Why Stark

- **Conversion up to 3.5× as fast as Lalamo in recorded benchmarks.** LFM2-350M MLX 4-bit took 0.532 s versus 1.853 s, a 71.3% lower complete-command median. [Measured comparison](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md).
- **92–95% less sampled conversion memory across Qwen3.5 and LFM2 MLX tests.** LFM2 4-bit used 113.0 MiB versus 1,411.4 MiB; Qwen3.5 8-bit used 203.5 MiB versus 4,155.3 MiB. These are sampled process-tree RSS measurements. [Memory evidence](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md).
- **85 of 110 pinned Lalamo profiles implemented, plus four Stark extensions.** Coverage includes 40 MLX profiles, hybrid/state-space models and ModernBERT sequence classifiers. Future compatible checkpoints can reuse architecture adapters.
- **Lightweight conversion tooling.** The installed converter needs no Lalamo, JAX or PyTorch. Transformations default to a bounded 8 MiB workspace, with full initial and final source verification.
- **Reference-checked exports.** Validated conversion tuples match pinned Lalamo tensor payloads, specifications, logical config and tokenizer metadata. Immutable source revisions and content hashes make each result traceable.
- **Exact probability readouts.** The scoring worker exposes raw candidate logits, log probabilities normalized over the full vocabulary, and temperature-calibrated probabilities over requested candidates.

## Recorded conversion benchmarks

All rows below target BF16 output and compare complete conversion commands against pinned Lalamo. Speed ratio is Lalamo median divided by Stark median; values above 1 favor Stark. Ratios are rounded from the displayed medians. Each row links to its source-bound checkpoint, converter revision, full ranges and independent review.

### Existing MLX 4/8-bit checkpoints

| Checkpoint / input | Stark median | Lalamo median | Speed ratio | Lower sampled RSS |
| --- | ---: | ---: | ---: | ---: |
| [LFM2-350M 4-bit](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md) | 0.532 s | 1.853 s | 3.48× | 92.0% |
| [LFM2-350M 8-bit](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md) | 0.623 s | 1.581 s | 2.54× | 92.6% |
| [Qwen3.5-0.8B text 4-bit](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md) | 0.997 s | 2.059 s | 2.07× | 94.9% |
| [Qwen3.5-0.8B text 8-bit](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md) | 1.393 s | 1.940 s | 1.39× | 95.1% |
| [Qwen3-0.6B 4-bit](docs/MODEL_CONVERSION_MILESTONE_3.md) | 1.011 s | 1.796 s | 1.78× | 93.0% |
| [Qwen3-0.6B 8-bit](docs/MODEL_CONVERSION_MILESTONE_3.md) | 1.506 s | 1.818 s | 1.21× | 93.3% |
| [Llama3.2-1B 4-bit](docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 1.256 s | 2.397 s | 1.91× | 94.2% |
| [Llama3.2-1B 8-bit](docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 2.202 s | 2.604 s | 1.18× | 96.6% |
| [SmolLM3-3B 4-bit](docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 2.779 s | 3.377 s | 1.22× | 95.4% |
| [SmolLM3-3B 8-bit](docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 5.083 s | 4.233 s | **0.83× (slower)** | 96.2% |
| [Gemma3-1B 4-bit](docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 1.551 s | 2.580 s | 1.66× | 90.7% |
| [Gemma3-1B 8-bit](docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) | 2.464 s | 2.537 s | 1.03× | 90.8% |
| [LLaMBA-1B 4-bit](docs/MODEL_CONVERSION_MILESTONE_5.md) | 2.142 s | 2.225 s | 1.04× | 96.4% |

The sequential coefficient writer also reduced conversion medians by **16–38% against the previous Stark implementation** across the six Llama, SmolLM3 and Gemma tuples, while preserving exact output bytes and the same workspace bound. [Writer improvement](docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md). MLX conversion repacks an already quantized checkpoint; it does not compute new quantization.

### Dense models and classifier fixtures

| Checkpoint / input | Stark median | Lalamo median | Speed ratio | Lower sampled RSS |
| --- | ---: | ---: | ---: | ---: |
| [LFM2-350M dense](docs/MODEL_CONVERSION_MILESTONE_4.md) | 1.233 s | 1.688 s | 1.37× | 90.2% |
| [Qwen3.5-0.8B text dense](docs/MODEL_CONVERSION_MILESTONE_4.md) | 3.525 s | 2.707 s | **0.77× (slower)** | 91.4% |
| [LLaMBA-1B dense F32 source](docs/MODEL_CONVERSION_MILESTONE_5.md) | 5.791 s | 2.105 s | **0.36× (slower)** | 97.1% |
| [ModernBERT BF16-source fixture](docs/MODEL_CONVERSION_MILESTONE_6.md) | 0.562 s | 1.137 s | 2.02× | 84.9% |
| [ModernBERT F32-source fixture](docs/MODEL_CONVERSION_MILESTONE_6.md) | 0.861 s | 1.312 s | 1.52× | 89.6% |

The two ModernBERT rows use unchanged public classifier weights with explicitly recorded synthetic chat-template overlays. All twenty classifier benchmark outputs passed full parity. These fixtures are separately identified from the two inaccessible Mirai catalog models. [Classifier fixture provenance](docs/MODEL_CONVERSION_MILESTONE_6.md).

**The remaining slow cases are visible:** SmolLM3 MLX 8-bit is 20.1% slower than Lalamo, dense Qwen3.5 is 30.2% slower, and dense LLaMBA takes 2.75× as long. Dense LLaMBA still reduces sampled RSS by 97.1%, from 5,817.6 MiB to 167.0 MiB. Smaller passing differences for Llama 8-bit, Gemma 8-bit and LLaMBA MLX have overlapping ranges, as do the ModernBERT F32 and dense LFM2 comparisons.

These are recorded warm-file measurements from successive independently reviewed converter revisions on one Apple-silicon host per packet. The initial Qwen3 packet uses three repeats per backend/tuple; the other displayed packets use five. Setup, downloads, warming, parity comparisons and native loading are outside command timing. Stark includes full source verification, output hashing and receipts; the Lalamo reference command has weaker source/receipt verification.

RSS sampling can miss peaks or double-count shared pages. The benchmarked 8 MiB bound applies to transformation buffers; headers, tokenizer processing, hashing and interpreter memory are additional. **These measurements establish conversion performance; inference latency and general generation quality require separate evaluation.** Earlier dense results and previous-writer measurements remain in the [evidence archive](docs/MODEL_CONVERSION_PLAN.md).

## Conversion coverage and future models

| Supported area | Current implementation |
| --- | --- |
| Dense language models | 43 built-in profiles: Qwen2/Qwen3/Qwen3.5 text, Llama, Mistral, Gemma text, Phi-4-mini, Granite, SmolLM, LFM2 and LLaMBA/Mamba2; four separately counted extensions include Gemma2 |
| Affine MLX 4/8-bit | 40 built-in profiles across Qwen3, Qwen3.5 text, Llama, SmolLM3, Gemma text, LFM2 and LLaMBA; supported bit widths and layouts remain profile-specific |
| Sequence classifiers | Two ModernBERT profiles, plus self-contained generic ModernBERT checkpoints using mean pooling and the supported GELU layout |
| Future checkpoints | `hf-generic` dispatches by architecture and input format, validates complete config/layout/metadata, and supports private offline source replay |

Compatible fine-tunes can use an existing adapter without a new catalog entry:

```sh
uv run --project tools/converter --locked stark-convert my-org/future-model \
  --metadata-profile hf-generic \
  --write-source-manifest runtime/converter-sources/future-model.json \
  --output models/future-model-stark
```

A generic checkpoint needs self-contained config/tokenizer metadata and a supported chat template. ModernBERT additionally requires `--model-kind classifier`; optional `--architecture` asserts the expected architecture. Known catalog identities retain their authoritative metadata and packing rules. See [supported families, formats and examples](tools/converter/README.md#future-checkpoints-and-architecture-tracking).

The catalog tracks **implementation**, while qualification tracks exact source/config/dtype/output/build tuples. **27 exact BF16 catalog tuples have named-build load-only qualifications; 62 implemented identities remain unqualified.** Generic outputs start native-unverified, and `--require-native-support` requires an exact stored catalog match. Classifier fixtures have separate private load-only proofs; both Mirai catalog repositories remain inaccessible to the current account and unqualified.

Twenty-five pinned built-ins remain planned, including specialized/MoE/multimodal and later-generation variants. Unsupported combinations fail explicitly. Complete Lalamo replacement remains an open goal. [Coverage and replacement plan](docs/MODEL_CONVERSION_PLAN.md).

## Quickstart

Run these commands from the repository root.

### Set up the converter

Install `uv`. It manages the converter's Python 3.13.13 environment:

```sh
uv sync --project tools/converter --locked
```

### Convert from Hugging Face

Pass the model's repository name. The converter downloads it and saves the result:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B \
  --output models/qwen3
```

### Convert a downloaded folder

Pass the path to the full model folder. This uses local files and needs no Hugging Face account:

```sh
uv run --project tools/converter --locked stark-convert /path/to/downloaded-model \
  --metadata-profile hf-generic --offline --output models/local-model
```

Use a new output folder each time. Your local folder must include the model config, tokenizer metadata and all safetensors weights. See [required files](tools/converter/README.md#convert-a-downloaded-folder) and [optional settings](tools/converter/README.md#optional-settings). Generic conversions remain native-unverified.

### Run the scoring worker

Use an Apple silicon Mac with Xcode's Metal tools and Rust 1.98.1. Build and start the worker with your converted model:

```sh
cargo build --release --locked -p stark-score --features optimized
cargo test --release --locked -p stark-score --features optimized -- --test-threads=1
./target/release/stark-score --score-model-path /absolute/path/to/model
```

The worker uses a private JSON-lines stdin/stdout protocol for tokenization and scoring. The supported standalone engine path is the scoring worker; host application integration belongs in separate repositories. [Scoring protocol](docs/SCORING.md).

## Scoring engine and validation

Stark adds direct scoring topology, deferred state initialization, fewer Metal submissions, final-layer pruning, compact delta convolution, reusable RoPE values and KV-only attention preparation. The `optimized` feature selects six accepted optional engine optimizations.

An opt-in `--consecutive-cache` reuses one completed internal vocabulary-logit entry for immediately consecutive identical token sequences with the same context size. Candidate selection and temperature can change because normalization and calibration are recomputed. Errors, reset and restart clear the cache; it defaults off and is never persisted.

Recorded JevK5 BF16 validation passed **243 evaluation cases / 486 cold-and-cache-hit replies**, with exact raw logits, log probabilities, calibrated candidate probabilities and tokenizations against frozen accepted Uzu scorer references. Calibration boundaries, full-vocabulary digests and process ownership have separate checks. This is source-bound scoring correctness evidence; precise engine latency benchmarking remains open. [Validation and build receipts](docs/VALIDATION.md).

The latest converter packet passed **3,672 tests**, with 60 private-reference checks skipped, plus formatting, locked dependency and repository-policy checks. Architecture packets have independent technical reviews and exact tensor/spec/config/tokenizer comparisons. Real model weights and operational artifacts remain ignored/private. Completed benchmark model files are removed after review; small receipts, observations, manifests, recipes and hashes are retained. [Classifier validation and cleanup](docs/MODEL_CONVERSION_MILESTONE_6.md).

## Scope, security and attribution

Native loading checks prove construction/readiness and EOF on the named build. Scoring, general generation and classifier execution have their own acceptance requirements. Inherited Python, TypeScript, Swift, CLI and playground binary distributions remain unqualified; no Stark packages or Swift XCFramework are published. Inherited cloud release and benchmark infrastructure stays disabled.

Native diagnostics and capture variables are suppressed. Protocol requests/responses belong in private pipes or files. Models, credentials, private evaluation data and runtime artifacts remain outside Git.

See [private security reporting](SECURITY.md), [CI and contribution checks](docs/CI.md), [rename compatibility](docs/RENAMING.md), [upstream provenance](provenance/UPSTREAM.md) and [LICENSE](LICENSE). Uzu and Mirai attribution is preserved.
