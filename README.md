<p align="center">
  <img src="docs/brand/stark-engine.jpg" width="200" alt="Stark: a charcoal cat with gold eyes">
</p>

<h1 align="center">Stark</h1>

<p align="center">
  <strong>Native inference on Apple silicon.</strong><br>
  Rust and Metal for local generation, probability scoring and model preparation.
</p>

<p align="center">
  <a href="#quickstart">Get started</a> ·
  <a href="#measured-performance">Benchmarks</a> ·
  <a href="docs/TECHNICAL_OVERVIEW.md">Technical guide</a> ·
  <a href="https://starkengine.app/">Website</a> ·
  <a href="website/README.md">Website source</a>
</p>

Stark is an experimental, open-source engine for applications that run models locally on a Mac. Its native workers expose a JSON-lines interface; your application controls the chat template, tools and user experience. Install the prebuilt Mac release below, or [build from source](docs/INSTALLATION.md#install-from-a-source-checkout).

| Generate | Score | Prepare |
| --- | --- | --- |
| Native text generation with fresh request state. | Candidate logits, full-vocabulary log probabilities and temperature-scaled candidate probabilities. | Streaming conversion of supported dense and existing MLX 4/8-bit checkpoints. |
| [Generation interface](docs/GENERATION.md) | [Scoring interface](docs/SCORING.md) | [Converter guide](tools/converter/README.md) |

## Measured performance

**7.36% higher end-to-end Qwen3.5-2B generation throughput than pinned MLX-LM** across four accepted paired runs on an Apple M5 Max with 40 GPU cores and 128 GiB of memory.

| Workload | Stark, tokens/s | MLX-LM, tokens/s | Stark advantage |
| --- | ---: | ---: | ---: |
| Fibonacci module | 358.2 | 321.5 | 11.40% |
| GPU explanation | 358.7 | 322.1 | 11.34% |
| Copy editing | 330.4 | 305.3 | 8.23% |
| Medium context | 315.0 | 296.7 | 6.13% |
| Long context | 238.8 | 230.6 | 3.53% |
| Long output | 344.2 | 331.6 | 3.83% |

This is a preliminary, single-model result at concurrency one. Both engines used the same affine W4/G64 checkpoint and input tokens, greedy sampling and fresh state. Rates include prefill, decode, IPC/readout and cleanup; loading and warmup are excluded. Each cell covers 20 requests. The 7.36% figure is the geometric mean of paired workload/load ratios.

Four of six planned pairs passed the frozen host guards; two were excluded for interference. Whole-campaign `timing_accepted` remains `false`. Answer-quality equivalence is unqualified, and HTTP serving, concurrent requests, other models and other Macs remain outside this result. [Read the short write-up](docs/STARK_BENCHMARK_BRIEF_2026_10_21.md) or inspect the [full protocol, versions, exclusions and quality results](docs/QWEN35_MLX_QUIET_COMPARISON_2026_10_20.md).

## Quickstart

Use an Apple silicon Mac with **macOS 26.4 or newer**. The [v0.1.1 signed release](https://github.com/jshahbazi/stark/releases/tag/v0.1.1) includes the workers, converter and a private Python runtime. Its native components are Developer ID signed and notarized by Apple. No source checkout, Rust, Xcode, uv or system Python setup is required.

```sh
curl -fL https://github.com/jshahbazi/stark/releases/download/v0.1.1/install.sh -o stark-install.sh
sh stark-install.sh --version 0.1.1
export PATH="$HOME/.local/bin:$PATH"
stark doctor
stark convert Qwen/Qwen3-0.6B --output models/qwen3
stark generate --model-path "$PWD/models/qwen3"
```

Installation defaults to `~/.local` without sudo and verifies the archive checksum before installing. Add `~/.local/bin` to your shell profile to keep it on PATH. See [installation, updates and uninstall](docs/INSTALLATION.md).

Use a new output directory for each conversion. The worker exposes `tokenize`, `decode` and `generate` over private stdin/stdout pipes. The caller renders the model's exact chat template. See [request examples and the generation contract](docs/GENERATION.md), or use `stark score --score-model-path /absolute/path/to/model` for decision models such as JevK5.

For downloaded or fine-tuned checkpoints, follow the [offline conversion and generic-model instructions](tools/converter/README.md#future-checkpoints-and-architecture-tracking). Generic outputs start native-unverified; check the exact model/configuration's qualification before relying on it.

## Evidence you can inspect

Stark keeps source revisions, build hashes, protocols, aggregate results and rejected attempts with its measurements.

- **Generation:** the [comparison inventory](docs/INFERENCE_COMPARISON_EVIDENCE.md) links stock engine, optimized Stark and predecessor-Uzu experiments. The [DeepInfer study](docs/DEEPINFER_ADOPTION_2026_10_21.md) retains two experiments that were reverted after demonstrating no useful gain.
- **Scoring:** a separate recorded JevK5 BF16 build passed **243 cases / 486 cold-and-cache-hit replies** against frozen accepted Uzu scorer references. [Validation scope and receipts](docs/VALIDATION.md)
- **Conversion:** a separate LFM2-350M MLX 4-bit study measured **0.532 s versus 1.853 s** for pinned Lalamo, with **92.0% lower sampled process-tree RSS** across five warm-file repeats. These are model-preparation measurements. [Study and full ranges](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md)

The [technical overview](docs/TECHNICAL_OVERVIEW.md) preserves the complete conversion tables, slower cases, coverage counts and validation boundaries. Implementation coverage and native qualification are tracked separately in the [model conversion plan](docs/MODEL_CONVERSION_PLAN.md). General production readiness and answer quality require application-specific validation.

## Project and provenance

Stark is a standalone fork of [Uzu](https://github.com/trymirai/uzu), starting at Uzu 0.6.1 (`8d01581474760bd1fc7f4a547ff1d02f7dada437`). Upstream Git history, copyright, MIT licensing and Mirai attribution are preserved. Host application integration belongs in separate repositories. Inherited cloud, signing and release infrastructure remains disabled; inherited binary distributions are unqualified.

Models, credentials, private evaluation data and runtime artifacts stay outside Git. Native protocol payloads belong in private pipes or files. See [security reporting](SECURITY.md), [CI and contribution checks](docs/CI.md), [upstream provenance](provenance/UPSTREAM.md), [rename compatibility](docs/RENAMING.md) and [LICENSE](LICENSE).
