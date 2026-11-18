# Qwen3.5 and Qwen3.8 checkpoints

Stark converts complete Qwen3.5-architecture checkpoints, including the official
Qwen3.8 dense model, into its native model format. Use BF16 output for native
scoring and text generation.

| Checkpoint family | Converter architecture | Execution path |
| --- | --- | --- |
| `Qwen/Qwen3-4B-Instruct-2507` | `qwen3` | Dense attention with query/key normalization |
| `Qwen/Qwen3.8-27B` | `qwen3_5` | Dense gated-DeltaNet/attention decoder with the existing Qwen3.5 vision family |
| `empero-ai/Qwen3.8-4B-Distill` | `qwen3_5` | Qwen3.5 text decoder with gated attention and DeltaNet |

The Qwen3.8 dense model declares the Transformers `qwen3_5` architecture. The
repository name does not select Stark's separate Qwen3.8 Flash Next MoE
implementation. Trained text-only exports declaring `model_type=qwen3_5_text`
use the same canonical `qwen3_5` adapter. Complete vision-language sources can
retain the reviewed image tower; conversion and execution qualification remain
artifact-specific.

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

The existing Qwen3 profile and complete Qwen3.5-family checkpoints support
direct conversion:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --output models/qwen3-4b-instruct-2507

uv run --project tools/converter --locked stark-convert empero-ai/Qwen3.8-4B-Distill \
  --revision c83cb7aa2999d2f35c43e9ae0634a30eb8985a1e \
  --output models/empero-qwen38-4b-distill

uv run --project tools/converter --locked stark-convert Qwen/Qwen3.8-27B \
  --revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0 \
  --artifact-mode vision-language \
  --output models/qwen38-27b
```

The pinned Qwen3.8 source is [the official config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/config.json), which declares Apache-2.0 licensing. Preserve its licensing and attribution when distributing weights. The exact 27B artifact still requires its own conversion, native-load, prefix-reuse, tool, image, memory and performance qualification.

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

Workers use the loaded model's configured context maximum: both public configs advertise 262,144 tokens. There is no separate 4,096-token generation cap. Omitted host output limits use the remaining finite context after tokenization; explicit limits are honored. The readiness reply reports the effective context, content/frame budgets and EOS IDs. Available memory and the [automatic shared resource budget](WORKER_PROTOCOLS.md#resource-budget) still bound individual requests; accepting the configured maximum does not qualify full-length execution.

## Validation boundaries

Converter regressions cover the pinned public config dimensions, tensor layouts, metadata and stop tokens, plus local trained-checkpoint routing with synthetic weights. The converter suite passes 3,793 tests, with 88 private-reference checks skipped. Ruff lint/format, locked dependencies and repository policy checks pass.

Native smoke checks of supplied trained archives are recorded privately, separately from the converter's load-only qualification ledger. The checks compare tokenization against the source tokenizer, repeat short greedy generation, test request-error recovery, and verify finite normalized scores and consecutive-cache recovery. These checks establish bounded functional use on the tested artifacts and workers. They do not establish Transformers output parity, output quality, maximum-context execution or compatibility with another training run. Private receipts retain archive/model/worker hashes and aggregate outcomes.

Generic trained checkpoints remain `native_loading=unverified` in their conversion receipts. `--require-native-support` intentionally rejects them because they have no exact entry in the named-build qualification ledger. Validate each trained checkpoint with your intended tests after conversion.
