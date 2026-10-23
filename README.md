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

Use an **Apple silicon Mac with macOS 26.4 or newer**, in a native arm64 terminal. The [latest signed release](https://github.com/jshahbazi/stark/releases/latest) includes terminal chat, the menu bar app, workers, converter and private Python runtimes. No source checkout or developer tools are required. Native components are Developer ID signed and notarized by Apple; the app includes its notarization ticket.

### Install and open the app

For drag-and-drop installation, download the signed, notarized `.dmg` from the [latest release](https://github.com/jshahbazi/stark/releases/latest), open it, and drag `Stark.app` into Applications. The script below installs both the app and terminal tools.

```sh
curl -fL https://github.com/jshahbazi/stark/releases/latest/download/install.sh -o stark-install.sh
sh stark-install.sh
export PATH="$HOME/.local/bin:$PATH"
stark doctor
open "$HOME/Applications/Stark.app"
```

The script selects the latest stable release automatically. Installation puts the CLI in `~/.local` and the app in `~/Applications`, without sudo. It verifies the archive checksum and stages both components before activation. Add `export PATH="$HOME/.local/bin:$PATH"` to `~/.zprofile` to keep the command available in new terminals. See [installation, updates and uninstall](docs/INSTALLATION.md).

`doctor` should report `Installation checks passed. No model was loaded.` In the menu bar, choose **Install Model**, select a supported local folder or Hugging Face repository, then **Run** to open chat. **Stop** unloads the worker. Models download separately. [Menu bar guide](apps/menu-bar/README.md)

### Use Stark in another app

Stark can run a local model for another desktop app on your Mac. Look for **OpenAI-compatible** or **Custom OpenAI** in that app's provider settings.

1. Open Stark, select a model and choose **Run**.
2. Open **Connection Details…** from Stark's menu or chat window.
3. In the other app, choose its OpenAI-compatible provider and paste the **Base URL**, **API key** and **Model ID** from that window. Save the settings and start a chat.

Keep Stark open with the model running. After reopening Stark, copy the connection details again: the API key changes, and the address may change. If you switch models, update the Model ID in the other app.

To try a request in Terminal, click **Copy curl Example** in Connection Details and paste the command into Terminal.

The API supports text chat and streaming. Browser-based clients and remote devices cannot connect. Tool calls, embeddings, image input and OpenAI's Responses API are not supported. See the [local API guide](docs/LOCAL_API.md) for the full compatibility details.

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

**Latest generation comparison — 2026-10-22:** Qwen3.5-2B (4-bit), Apple M5 Max (40 GPU cores, 128 GiB), one request at a time.

| Engine / configuration | Build / version | End-to-end tokens/s |
| --- | --- | ---: |
| Stark | `97f15f38` | **245.5** |
| MLX-LM | `72be8c955` / MLX 0.32.4 | 218.8 |
| vLLM Metal | 0.32.0 | 174.7 |
| llama.cpp | b11747 | 147.9 |
| TensorFold (drafting on) | 1.1.1 | 125.6 |
| TensorFold (drafting off) | 1.1.1 | 118.5 |

**Stark recorded 12.2% higher observed aggregate throughput than MLX-LM** and led all six workloads in this run. All engines were measured afresh: four independent loads per configuration and three repeats per workload, totaling 432 performance requests.

Rates include input processing, generation, transport/readout and request cleanup; loading and warmup are excluded. The aggregate is the geometric mean of six workload rates, each based on four per-load medians. These are observed desktop results; performance varies with system load, model and hardware.

llama.cpp uses a separate Q4_K_M quantizer; the other configurations use affine W4/G64. MetalRT remains unmeasured because stock support for this exact model is unverified.

[Interactive comparison](https://starkengine.app/#benchmarks) · [Method, versions and quality checks](https://github.com/jshahbazi/stark/blob/b64daf27015d346761e369885ea320b55222b846/docs/QWEN35_ENGINE_COMPARISON_2026_10_22.md) · [Machine-readable results](https://raw.githubusercontent.com/jshahbazi/stark/b64daf27015d346761e369885ea320b55222b846/website/public/benchmark-results.json)

Separate source-bound studies record **243 JevK5 cases / 486 cold-and-cache-hit scoring replies** matching frozen Uzu references ([validation](docs/VALIDATION.md)), and **0.532 s versus 1.853 s** conversion time with **92.0% lower sampled process-tree RSS** for LFM2-350M MLX 4-bit across five warm-file repeats ([conversion study](docs/MODEL_CONVERSION_MILESTONE_4_MLX.md)).

The [evidence inventory](docs/INFERENCE_COMPARISON_EVIDENCE.md) and [technical overview](docs/TECHNICAL_OVERVIEW.md) retain source revisions, hashes, slower cases, rejected attempts and qualification limits. The [DeepInfer study](docs/DEEPINFER_ADOPTION_2026_10_21.md) retains two reverted experiments. These results apply to their recorded builds; production readiness and answer quality require application-specific validation.

## Project and provenance

Stark is a standalone fork of [Uzu](https://github.com/trymirai/uzu), starting at Uzu 0.6.1 (`8d01581474760bd1fc7f4a547ff1d02f7dada437`). Upstream Git history, copyright, MIT licensing and Mirai attribution are preserved; inherited binary distributions remain unqualified.

See [security reporting](SECURITY.md), [upstream provenance](provenance/UPSTREAM.md), [rename compatibility](docs/RENAMING.md) and [LICENSE](LICENSE).
