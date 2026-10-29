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
  <a href="#supported-model-families">Models</a> ·
  <a href="#measured-performance">Benchmarks</a> ·
  <a href="#developing-stark">Developers</a> ·
  <a href="docs/README.md">Documentation</a> ·
  <a href="https://starkengine.app/">Website</a>
</p>

Stark is an experimental, open-source Rust/Metal engine for running language models locally on Apple silicon. It provides a menu bar app, terminal chat, a loopback OpenAI-style API, probability scoring and model conversion. Applications can also drive its native workers directly through private stdin/stdout pipes with one JSON request and reply per line.

| Generate | Score | Prepare |
| --- | --- | --- |
| Terminal chat and native text generation with fresh state per request. | Candidate logits, full-vocabulary log probabilities and temperature-scaled candidate probabilities. | Streaming conversion of supported dense and existing MLX 4/8-bit checkpoints. |
| [Generation protocol](docs/GENERATION.md) | [Scoring protocol](docs/SCORING.md) | [Converter guide](tools/converter/README.md) |

## Quickstart

Use an **Apple silicon Mac with macOS 26.4 or newer** in a native arm64 terminal. The [latest signed release](https://github.com/jshahbazi/stark/releases/latest) includes the menu bar app, terminal chat, both workers, the converter and private Python runtimes; no source checkout or developer tools are needed.

### Install the app

Download the signed, notarized `.dmg` from the [latest release](https://github.com/jshahbazi/stark/releases/latest) and open **Install Stark** when included. It replaces the running copy and launches the new app. Older DMGs use drag-and-drop: copy `Stark.app` into Applications, quit the old app, then open the installed copy. Click the cat icon and choose **Manage Models**. Use **Model Catalog** or **Add from Source** to add a model, choose **Run** in **Installed**, then **Open Playground** to chat. [Menu bar guide](apps/menu-bar/README.md)

### Install the app and terminal tools

```sh
curl -fL https://github.com/jshahbazi/stark/releases/latest/download/install.sh -o stark-install.sh
sh stark-install.sh
export PATH="$HOME/.local/bin:$PATH"
stark doctor
open "$HOME/Applications/Stark.app"
```

The script installs the CLI under `~/.local` and the app under `~/Applications` without sudo and checks the archive checksum before activating anything. Add the `PATH` line to `~/.zprofile` to keep `stark` available. The DMG and the in-app updater verify code signatures; the script checks a checksum from the same release. See [installation](docs/INSTALLATION.md), including [what each route verifies](docs/INSTALLATION.md#what-each-install-route-verifies).

### Chat in Terminal

```sh
stark convert Qwen/Qwen3-0.6B --output models/qwen3
stark chat --model-path "$PWD/models/qwen3"
```

Conversion downloads the model; use a new output directory each time. Chat loads it: one message per line, `/reset` clears history, `/exit` quits. See [chat options](docs/GENERATION.md#interactive-cli-chat).

### Use Stark from another app

Stark serves an OpenAI-style chat API for clients on the same Mac. Run a model, open **Connection Details** from Stark's menu, and paste the **Base URL**, **API key** and **Model ID** into the client's OpenAI-compatible provider settings. The key changes on every launch. Browser-origin requests, remote devices, tool calls, embeddings and structured output are unsupported. See the [local API contract](docs/LOCAL_API.md) and [tested clients](docs/LOCAL_API.md#client-compatibility).

The [native image path](docs/MULTIMODAL_IMPLEMENTATION.md) accepts one PNG/JPEG through the CLI, Local API, and Swift attachments. Compatible dense Qwen3.5 BF16 models and merged fine-tunes use the [configuration-driven family contract](docs/VISION_FAMILY_CONTRACT.md), with automatic local checks before image input is enabled. Reference qualification is separately recorded for an exact 0.8B artifact, worker, device and macOS tuple. The app supports attachment selection, drop, paste, preview, removal, and image conversations.

## Models and application integration

### Supported model families

The tables below include named checkpoints and fine-tunes as well as their implementation families. Braces abbreviate size or version choices, such as `Qwen3-{0.6B,1.7B}`.

#### Decision models

| Model | Supported use | Images |
| --- | --- | --- |
| **JevK5** (`alibiserikbay/JevK5`, BF16 Qwen3.5-4B) | Candidate probability scoring and the app's Decision workflow. [Recorded scoring parity](docs/validation/SCORING_PARITY.md) covers 243 cases on its named build; [chat EOS checks](docs/TERMINATOR_METRICS_VALIDATION_2026_10_25.md) also cover native generation. | The recorded JevK5 artifact is text-only. |

The [scoring worker](docs/SCORING.md) can also score compatible language models below. Decision use requires a direct-answer template, distinct single-token labels and checkpoint-specific checks. JevK5's recorded parity applies to its exact checkpoint and build; decision accuracy and calibration need their own evaluation.

#### Text models

| Family / adapter | Named models and checkpoint coverage |
| --- | --- |
| Qwen3 dense | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-4B-Thinking-2507`, **Polaris-4B-Preview** (`POLARIS-Project/Polaris-4B-Preview`) |
| Qwen2 dense | **Qwen2.5-Coder** (`Qwen/Qwen2.5-Coder-{0.5B,1.5B,3B,7B,14B,32B}-Instruct`), **DeepSeek-R1-Distill-Qwen-1.5B** (`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) |
| Qwen3.5 dense | `Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B}`, **Qwen3.8-4B-Distill** (`empero-ai/Qwen3.8-4B-Distill`), **JevK5** (see Decision models above) |
| Llama dense | `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-{1B,3B}-Instruct`, `unsloth/Llama-3.2-1B-Instruct`, **SmolLM2** (`HuggingFaceTB/SmolLM2-1.7B-Instruct`), **Nanbeige4.1** (`Nanbeige/Nanbeige4.1-3B`), **Reka Flash 3.1** (`RekaAI/reka-flash-3.1`) |
| Mistral dense | **Mistral 7B Instruct** (`mistralai/Mistral-7B-Instruct-{v0.2,v0.3}`), **Codestral** (`mistralai/Codestral-22B-v0.1`), **Devstral Small** (`mistralai/Devstral-Small-2505`) |
| Gemma 2 | `google/gemma-2-2b-it` |
| Gemma 3 text | `google/gemma-3-1b-it`, **FunctionGemma** (`google/functiongemma-270m-it`), **rnj-1** (`EssentialAI/rnj-1-instruct`) |
| Phi-3 layout | **Phi-4-mini-instruct** (`microsoft/Phi-4-mini-instruct`); supported tied-embedding layout with LongRoPE |
| Granite dense | `ibm-granite/granite-{3.1,3.3}-{2b,8b}-instruct` |
| SmolLM3 | `HuggingFaceTB/SmolLM3-3B` |
| LFM2 | `LiquidAI/LFM2-{350M,700M,1.2B,2.6B}` |
| LLaMBA / Mamba2 | `cartesia-ai/Llamba-1B`, `cartesia-ai/Llamba-1B-4bit-mlx` |

These converter paths accept supported floating Safetensors layouts and produce native BF16 artifacts. Existing MLX affine 4/8-bit exports are also supported for Qwen3, Qwen3.5, Llama, Gemma 3 text, SmolLM3, LFM2 and LLaMBA, subject to each adapter's layout and group-size constraints. The [converter coverage table](tools/converter/README.md#catalog-coverage-and-qualification) lists the registered export names. Compatible complete checkpoints and merged fine-tunes can use generic conversion even when absent from the catalog.

#### Native original-format models

| Model | Prepared checkpoint choices |
| --- | --- |
| **GLM-5.3-Flash** | `unsloth/GLM-5.3-Flash-GGUF` (`UD-Q2_K_XL`) |
| **Qwen3.8-Flash-Next** | `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` |
| **Recursive Flint** | `RecursiveSI/flint-int4` |

These use separate Rust/Metal execution paths and require the [pinned preparation commands](docs/NATIVE_HYBRID_MODELS.md). See [Flint setup and limits](docs/FLINT.md). Their current Stark paths accept text input.

#### Image models

| Family / model | Image support in Stark |
| --- | --- |
| **Qwen3.5 dense BF16** | Full vision-language conversion for compatible complete checkpoints and merged fine-tunes. The family contract's configuration fixtures cover 0.8B, 2B, 4B, 9B and 27B dimensions; each artifact must pass native admission and automatic local image checks. |
| **Qwen3.5-0.8B BF16** | Also has [recorded reference parity and image execution evidence](docs/MULTIMODAL_IMPLEMENTATION.md), scoped to the recorded artifact, worker, device and macOS version. |
| All other families and the recorded text-only JevK5 artifact | Image input is currently unsupported. This includes Gemma 3 multimodal variants, Qwen3.5 MoE, GLM-5.3-Flash, Qwen3.8-Flash-Next and Flint. |

Image input supports one PNG/JPEG through Playground, terminal chat and the Local API. Convert with `--artifact-mode vision-language`; the default text conversion omits vision tensors. Quantized vision weights and LoRA-only adapters are unsupported. A fine-tune's name or base family alone does not establish image support: it needs the complete supported vision tower, BF16 weights and reviewed tokenizer, template and processor metadata. See the [vision family contract](docs/VISION_FAMILY_CONTRACT.md).

#### Classifier models

**ModernBERT** conversion covers `trymirai/chat-moderation-router` and `trymirai/health-related-router`; the [converter guide](tools/converter/README.md#catalog-coverage-and-qualification) records both repositories as inaccessible. These produce classifier artifacts; the app has no ModernBERT chat, Decision or image workflow.

Family support describes implemented code paths. Each checkpoint needs compatible metadata and tensors and a successful native load; quality, throughput and Decision calibration require separate validation. These are source-tree capabilities; installed releases may contain an earlier set of families.

The [converter guide](tools/converter/README.md#catalog-coverage-and-qualification) details formats and checkpoint coverage. Existing supported MLX 4/8-bit weights are repacked without requantization. For a complete downloaded or merged fine-tuned checkpoint:

```sh
stark convert /absolute/path/to/checkpoint --metadata-profile hf-generic \
  --offline --output models/local-model
```

The folder needs complete safetensors weights, config, tokenizer, chat template and end-of-sequence metadata; merge LoRA/PEFT adapters first. A successful conversion checks the layout, not inference quality. See [future checkpoints](tools/converter/README.md#future-checkpoints-and-architecture-tracking).

Application authors can drive a worker directly:

| Task | Command |
| --- | --- |
| Generate text | `stark generate --model-path /absolute/path/to/converted-model` |
| Score candidates | `stark score --score-model-path /absolute/path/to/converted-model` |

Both wait for JSON-lines requests. The application renders the chat template and owns tools and the agent loop. See [worker protocols](docs/WORKER_PROTOCOLS.md). Host integrations belong in separate repositories.

## Developing Stark

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before changing code. Source builds need Xcode's Metal Toolchain, rustup and uv; commands run from the repository root with Rust **1.98.1**, uv **0.12.24** and locked dependencies.

| Area | Code and guide |
| --- | --- |
| Engine and Metal kernels | `crates/stark-engine/`; [technical guide](docs/TECHNICAL_OVERVIEW.md) |
| Generation and scoring workers | `crates/stark-generate/`, `crates/stark-score/`; [worker protocols](docs/WORKER_PROTOCOLS.md) |
| Converter, chat, local API and app backend | `tools/converter/stark_convert/`; [converter guide](tools/converter/README.md), [local API](docs/LOCAL_API.md) |
| Menu bar app | `apps/menu-bar/`; [build and usage](apps/menu-bar/README.md) |
| CLI launcher, installer, release | `scripts/stark`, `install.sh`, `scripts/`; [installation](docs/INSTALLATION.md), [releasing](docs/RELEASING.md) |
| Checks and validation | [CI](docs/CI.md), [validation records](docs/validation/), [benchmarks](docs/benchmarks/README.md) |
| Website | `website/`; [setup and publishing](website/README.md) |

Build the workers with `cargo +1.98.1 build --release --locked -p stark-generate` and, separately, `cargo +1.98.1 build --release --locked -p stark-score --features optimized`. For checkout chat use `./scripts/stark chat --model-path /absolute/path/to/converted-model`. Start with the repository policy check, then the [checks for the area you changed](docs/CI.md#local-checks):

```sh
uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py
```

The inherited Uzu bindings, server and command-line crates have been removed (see [upstream provenance](provenance/UPSTREAM.md#removed-inherited-code)); the inherited cloud, signing and release workflows stay disabled.

## Measured performance

Qwen3.5-2B (4-bit), Apple M5 Max (40 GPU cores, 128 GiB), one request at a time, measured 2026-10-22:

| Engine | Build / version | End-to-end tokens/s |
| --- | --- | ---: |
| Stark | `97f15f38` | **245.5** |
| MLX-LM | `72be8c955` / MLX 0.32.4 | 218.8 |
| vLLM Metal | 0.32.0 | 174.7 |
| llama.cpp | b11747 | 147.9 |
| TensorFold (drafting on) | 1.1.1 | 125.6 |
| TensorFold (drafting off) | 1.1.1 | 118.5 |

Stark led all six workloads with 12.2% higher aggregate throughput than MLX-LM. Rates include input processing, generation, transport and cleanup, and exclude loading and warmup; the aggregate is the geometric mean of six workload rates from four loads and three repeats each. Results vary with system load, model and hardware, and apply to the recorded builds. [Interactive comparison](https://starkengine.app/#benchmarks) · [Method and versions](docs/benchmarks/QWEN35_ENGINE_COMPARISON_2026_10_22.md) · [All benchmarks](docs/benchmarks/README.md) · [Conversion benchmarks](docs/benchmarks/CONVERSION_BENCHMARKS.md)

## Project and provenance

Stark is a standalone fork of [Uzu](https://github.com/trymirai/uzu) starting at Uzu 0.6.1 (`8d01581474760bd1fc7f4a547ff1d02f7dada437`). Upstream Git history, copyright, MIT licensing and Mirai attribution are preserved. See [security policy](SECURITY.md), [upstream provenance](provenance/UPSTREAM.md) and [LICENSE](LICENSE).
