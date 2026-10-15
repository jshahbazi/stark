# Qwen 4B trained checkpoints

Stark converts full or merged checkpoints derived from `Qwen/Qwen3-4B-Instruct-2507` and `empero-ai/Qwen3.8-4B-Distill` into its native model format. Use BF16 output for native scoring and text generation.

| Checkpoint family | Converter architecture | Execution path |
| --- | --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | `qwen3` | Dense attention with query/key normalization |
| `empero-ai/Qwen3.8-4B-Distill` | `qwen3_5` | Qwen3.5 text decoder with gated attention and DeltaNet |

The Empero model uses the Qwen3.5 architecture. Its repository name does not select a separate Qwen3.8 architecture. Trained text-only exports declaring `model_type=qwen3_5_text` use the same canonical `qwen3_5` adapter. Stark also extracts the text decoder from a declared vision-language wrapper and source-verifies omitted vision/MTP weights. Image, video and speculative MTP execution are outside this path.

## Convert your trained checkpoint

Keep each checkpoint in its own source directory with:

- `config.json` describing the trained model;
- all Safetensors weight files and their index when sharded;
- `tokenizer.json` and `tokenizer_config.json` from training;
- the exact chat template, inline in `tokenizer_config.json` or in `chat_template.jinja`;
- explicit EOS IDs in `config.json` or `generation_config.json`.

Preserve `generation_config.json` when available. When both template representations are present, they must agree exactly. A tokenizer changed during training must accompany its corresponding embedding weights.

For a Qwen3-4B-Instruct-2507 checkpoint:

```sh
uv run --project tools/converter --locked stark-convert /absolute/path/to/qwen3-checkpoint \
  --metadata-profile hf-generic --architecture qwen3 --offline \
  --write-source-manifest runtime/converter-sources/trained-qwen3.json \
  --output models/trained-qwen3
```

For an Empero Qwen3.8-4B-Distill checkpoint:

```sh
uv run --project tools/converter --locked stark-convert /absolute/path/to/empero-checkpoint \
  --metadata-profile hf-generic --architecture qwen3_5 --offline \
  --write-source-manifest runtime/converter-sources/trained-empero.json \
  --output models/trained-empero
```

These commands use each checkpoint's own metadata. `--architecture` checks the declared architecture and rejects a mismatch. Use a fresh output directory for every conversion. The default output dtype is BF16; F16/F32 exports remain conversion-only. Existing supported MLX affine 4/8-bit full checkpoints can use the same generic route when their quantization declarations and complete matrix planes pass validation.

Qwen3 exports may declare unscaled RoPE with legacy `rope_theta` or modern `rope_parameters` containing `rope_type=default` and `rope_theta`. Conflicting declarations and scaled/custom variants are rejected.

An adapter-only LoRA/PEFT directory is insufficient. Merge its adapters into the exact base revision using the training toolkit, then export a complete Safetensors model and its tokenizer/config metadata before conversion. Stark does not apply training adapters at runtime. Pickle `.bin` training checkpoints and optimizer/trainer state are not accepted model inputs.

For a private Hugging Face checkpoint, replace the local source path with its repository identity, remove `--offline`, and pin `--revision` to its immutable commit. Use existing Hugging Face credential storage. Model files, private manifests and conversion receipts remain ignored; preserve them locally for replay.

## Convert the public base models

The existing Qwen3 profile and the Empero extension support direct conversion:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --output models/qwen3-4b-instruct-2507

uv run --project tools/converter --locked stark-convert empero-ai/Qwen3.8-4B-Distill \
  --revision c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e \
  --output models/empero-qwen38-4b-distill
```

The pinned public sources are [Qwen's config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json) and [Empero's config](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill/blob/c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e/config.json). Both checkpoints declare Apache-2.0 licensing; preserve their licensing and attribution when distributing weights.

## Run tests through Stark

Build on Apple silicon with Rust 1.98.1 and the Xcode Metal tools:

```sh
cargo build --release --locked -p stark-score -p stark-generate
./target/release/stark-score --score-model-path /absolute/path/to/models/trained-qwen3
```

Use the [scoring protocol](SCORING.md) to tokenize a rendered input and score candidate tokens. For text generation, start the generation worker instead:

```sh
./target/release/stark-generate --model-path /absolute/path/to/models/trained-empero
```

The caller renders the exact training chat template and passes the result through the [generation protocol](GENERATION.md#private-json-lines-protocol). Preserve the checkpoint's thinking behavior; generic conversion supplies no reasoning control or output parser. Keep prompt/response text, token vectors and decision probabilities inside private pipes. The generation worker preserves reasoning text in its reply, so the caller owns its interpretation.

Current workers cap context at 16,384 tokens, even though both public configs advertise 262,144. Generation is additionally capped at 4,096 new tokens. The readiness reply reports the effective limits and EOS IDs.

## Validation boundaries

Converter regressions cover the pinned public config dimensions, tensor layouts, metadata and stop tokens, plus local trained-checkpoint routing with synthetic weights. The converter suite passes 3,793 tests, with 88 private-reference checks skipped. Ruff lint/format, locked dependencies and repository policy checks pass.

Native smoke checks of supplied trained archives are recorded privately, separately from the converter's load-only qualification ledger. The checks compare tokenization against the source tokenizer, repeat short greedy generation, test request-error recovery, and verify finite normalized scores and consecutive-cache recovery. These checks establish bounded functional use on the tested artifacts and workers. They do not establish Transformers output parity, output quality, maximum-context execution or compatibility with another training run. Private receipts retain archive/model/worker hashes and aggregate outcomes.

Generic trained checkpoints remain `native_loading=unverified` in their conversion receipts. `--require-native-support` intentionally rejects them because they have no exact entry in the named-build qualification ledger. Validate each trained checkpoint with your intended tests after conversion.
