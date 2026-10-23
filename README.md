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
  <a href="#developing-stark">Developers and agents</a> ·
  <a href="docs/TECHNICAL_OVERVIEW.md">Technical guide</a> ·
  <a href="https://starkengine.app/">Website</a>
</p>

Stark is an experimental, open-source Rust/Metal engine for running language models locally on Apple silicon. It provides terminal chat, a native menu bar app, probability scoring and model conversion. Applications can use its native workers through private stdin/stdout pipes with one JSON request and reply per line.

| Generate | Score | Prepare |
| --- | --- | --- |
| Terminal chat and native text generation with fresh request state. | Candidate logits, full-vocabulary log probabilities and temperature-scaled candidate probabilities. | Streaming conversion of supported dense and existing MLX 4/8-bit checkpoints. |
| [Generation interface](docs/GENERATION.md) | [Scoring interface](docs/SCORING.md) | [Converter guide](tools/converter/README.md) |

## Quickstart

Use an **Apple silicon Mac with macOS 26.4 or newer**, in a native arm64 terminal. The [v0.1.2 signed release](https://github.com/jshahbazi/stark/releases/tag/v0.1.2) includes terminal chat, the menu bar app, workers, converter and private Python runtimes. No source checkout or developer tools are required. Native components are Developer ID signed and notarized by Apple; the app includes its notarization ticket.

### Install and open the app

```sh
curl -fL https://github.com/jshahbazi/stark/releases/download/v0.1.2/install.sh -o stark-install.sh
sh stark-install.sh --version 0.1.2
export PATH="$HOME/.local/bin:$PATH"
stark doctor
open "$HOME/Applications/Stark.app"
```

Installation puts the CLI in `~/.local` and the app in `~/Applications`, without sudo. It verifies the archive checksum and stages both components before activation. Add `export PATH="$HOME/.local/bin:$PATH"` to `~/.zprofile` to keep the command available in new terminals. See [installation, updates and uninstall](docs/INSTALLATION.md).

`doctor` should report `Installation checks passed. No model was loaded.` In the menu bar, choose **Install Model**, select a supported local folder or Hugging Face repository, then **Run** to open chat. **Stop** unloads the worker. Models download separately. [Menu bar guide](apps/menu-bar/README.md)

### Chat in Terminal

```sh
stark convert Qwen/Qwen3-0.6B --output models/qwen3
stark chat --model-path "$PWD/models/qwen3"
```

Conversion downloads the model; use a new output directory each time. Chat loads it: type one message per line, use `/reset` to clear history and `/exit` to quit. Replies appear after generation completes; conversation history stays in memory. See [chat options](docs/GENERATION.md#interactive-cli-chat).

To build Stark yourself, follow the [source installation guide](docs/INSTALLATION.md#install-from-a-source-checkout). [Unsigned development bundles](docs/CI.md#download-a-development-build) are also available from successful Actions **Build** runs; check the bundle's source commit.

## Models and application integration

The [converter guide](tools/converter/README.md#catalog-coverage-and-qualification) lists supported families, layouts and exact qualification records. Conversion covers selected Qwen, Llama, Gemma, LFM2 and other architectures; existing MLX 4/8-bit conversion preserves quantization. Each model's weights must fit in available memory.

For a complete downloaded or merged fine-tuned checkpoint, convert offline:

```sh
stark convert /absolute/path/to/checkpoint --metadata-profile hf-generic \
  --offline --output models/local-model
```

The folder needs complete Safetensors weights, config, tokenizer, chat template and end-of-sequence metadata. Merge LoRA/PEFT adapters first. Gated online downloads require your existing Hugging Face access. Generic conversions start **native-unverified**; a successful conversion does not establish loading, inference correctness or answer quality. See [checkpoint requirements](tools/converter/README.md#future-checkpoints-and-architecture-tracking).

Application authors can choose a worker for a converted model:

| Task | Command |
| --- | --- |
| Generate text | `stark generate --model-path /absolute/path/to/converted-model` |
| Score candidates | `stark score --score-model-path /absolute/path/to/converted-model` |

These commands wait for JSON-lines requests. `generate` exposes `tokenize`, `decode` and `generate`; `score` provides probability readouts for decision models such as JevK5. Follow the [generation](docs/GENERATION.md#private-json-lines-protocol) or [scoring](docs/SCORING.md) contract. Applications own template rendering, tools and the agent loop; host integrations belong in separate repositories.

## Developing Stark

Read [AGENTS.md](AGENTS.md) before changing code. Source builds require Xcode's Metal Toolchain and rustup. Run commands from the repository root with Rust **1.98.1**, uv **0.12.24** and locked dependencies; the converter uses Python **3.13.13**. Keep models, binaries, credentials and private evaluation/runtime artifacts outside Git. Preserve validation records against their exact source and build; record new runs separately.

| Area | Code and current guidance |
| --- | --- |
| Native engine and Metal kernels | `crates/stark-engine/`; [technical overview](docs/TECHNICAL_OVERVIEW.md) |
| Generation and scoring workers | `crates/stark-generate/`, `crates/stark-score/`; [generation](docs/GENERATION.md), [scoring](docs/SCORING.md) |
| CLI, chat and conversion | `scripts/stark`, `tools/converter/stark_convert/`; [converter guide](tools/converter/README.md) |
| Menu bar application | `apps/menu-bar/`; [build and usage](apps/menu-bar/README.md) |
| Checks and validation | [CI commands](docs/CI.md#local-checks), [scoring evidence](docs/VALIDATION.md), [conversion status and remaining work](docs/MODEL_CONVERSION_PLAN.md) |
| Website | `website/`; [local setup and publishing](website/README.md) |

Build the generation worker directly with `cargo +1.98.1 build --release --locked -p stark-generate`. Keep the scorer's optimized build separate: `cargo +1.98.1 build --release --locked -p stark-score --features optimized`. For checkout chat, use `./scripts/stark chat --model-path /absolute/path/to/converted-model`; the launcher uses the built worker and locked converter environment.

Start validation with the repository policy check (its separate environment uses Python 3.12.13), then run the [checks for the area you changed](docs/CI.md#local-checks):

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
```

`crates/legacy/` contains inherited Uzu applications and bindings. The active standalone path is the engine, workers and converter listed above. Inherited cloud, signing and release workflows remain disabled; Stark's separate distribution workflows are documented in [release operations](docs/RELEASING.md).

## Measured performance

**7.36% higher end-to-end Qwen3.5-2B generation throughput than pinned MLX-LM** across four accepted paired runs on an Apple M5 Max with 40 GPU cores and 128 GiB of memory.

This preliminary result uses one model at concurrency one, the same affine W4/G64 checkpoint and input tokens, greedy sampling and fresh state. Rates include prefill, decode, IPC/readout and cleanup; loading and warmup are excluded. The figure is the geometric mean of paired workload/load ratios across six workloads, with 20 requests per engine/workload.

Four of six planned pairs passed the frozen host guards; two were excluded for interference. Whole-campaign `timing_accepted` remains `false`. Answer-quality equivalence is unqualified; HTTP serving, concurrent requests, other models and other Macs remain outside this result. [Short write-up](docs/STARK_BENCHMARK_BRIEF_2026_10_21.md) · [full protocol, versions, workload table and exclusions](docs/QWEN35_MLX_QUIET_COMPARISON_2026_10_20.md)

Separate source-bound studies record **243 JevK5 cases / 486 cold-and-cache-hit scoring replies** matching frozen Uzu references ([validation](docs/VALIDATION.md)), and **0.532 s versus 1.853 s** conversion time with **92.0% lower sampled process-tree RSS** for LFM2-350M MLX 4-bit across five warm-file repeats ([conversion study](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md)).

The [evidence inventory](docs/INFERENCE_COMPARISON_EVIDENCE.md) and [technical overview](docs/TECHNICAL_OVERVIEW.md) retain source revisions, hashes, slower cases, rejected attempts and qualification limits. The [DeepInfer study](docs/DEEPINFER_ADOPTION_2026_10_21.md) retains two reverted experiments. These results apply to their recorded builds; production readiness and answer quality require application-specific validation.

## Project and provenance

Stark is a standalone fork of [Uzu](https://github.com/trymirai/uzu), starting at Uzu 0.6.1 (`8d01581474760bd1fc7f4a547ff1d02f7dada437`). Upstream Git history, copyright, MIT licensing and Mirai attribution are preserved; inherited binary distributions remain unqualified.

See [security reporting](SECURITY.md), [upstream provenance](provenance/UPSTREAM.md), [rename compatibility](docs/RENAMING.md) and [LICENSE](LICENSE).
