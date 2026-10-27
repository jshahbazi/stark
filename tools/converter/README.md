# Stark model converter

`stark-convert` turns supported Hugging Face checkpoints into the `config.json`, `tokenizer.json` and `model.safetensors` files the Stark engine loads. It streams tensors through a bounded workspace, verifies the source before and after conversion, and writes a `conversion.json` receipt next to the output. The same Python package also contains the terminal chat client and the menu bar app's backend; this guide covers conversion.

## Set up

An installed Stark bundle provides `stark convert` with its own Python runtime; see [installation](../../docs/INSTALLATION.md). To work in the source checkout, install `uv` and run from the repository root:

```sh
uv sync --project tools/converter --locked
```

The converter uses Python 3.13.13. Every example below uses `uv run --project tools/converter --locked stark-convert`; with an installed bundle, replace that prefix with `stark convert`.

## Convert from Hugging Face

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B --output models/qwen3
```

The source may be a repository name or its page URL (`https://huggingface.co/Qwen/Qwen3-0.6B`, with or without a trailing slash, query or fragment). The converter resolves the revision to an immutable commit, downloads only the files it needs, and writes the result to the output directory, which must not exist yet. Gated repositories use your existing Hugging Face credentials. Use `--revision` to pin a branch, tag or commit.

## Convert a downloaded folder

```sh
uv run --project tools/converter --locked stark-convert /path/to/downloaded-model \
  --metadata-profile hf-generic --offline --output models/local-model
```

Give the path with a leading `/`, `./` or `~`; a bare name that is not an existing directory is treated as a Hugging Face repository. The folder must contain `config.json`, `tokenizer.json`, `tokenizer_config.json`, every `.safetensors` shard and the index file when the weights are sharded. Language models also need a chat template (`chat_template.jinja`, or the template inside `tokenizer_config.json`; a separate `chat_template.json` is not recognised) and non-empty end-of-sequence token IDs in `config.json` or `generation_config.json`. Merge LoRA/PEFT adapters into a complete checkpoint first.

For a downloaded copy of a [catalog](#catalog-coverage-and-qualification) model, use its registered profile instead of `hf-generic` so that the catalog's metadata rules apply:

```sh
uv run --project tools/converter --locked stark-convert /path/to/downloaded-model \
  --repo Qwen/Qwen3-0.6B --offline --output models/local-qwen3
```

Some catalog profiles take tokenizer or generation metadata from another repository; those need a [source manifest](#offline-replay-and-source-manifests) for offline conversion.

## Output

The output directory contains `config.json`, `tokenizer.json`, `model.safetensors` and `conversion.json`. The receipt records the source revision and content hashes, converter inputs, dependency lock, emitted file hashes, workspace use and the native-loading status. BF16 is the default output dtype and the only one the engine loads; F16 and F32 outputs are conversion-only and are marked runtime-blocked. The converter stages into a private temporary directory and publishes with an atomic no-replace rename, so a failed or interrupted conversion leaves no partial output.

A successful conversion validates the complete supported layout. It does not run the model. Text conversions report `native_loading: "unverified"` unless the exact artifact matches a stored qualification record. Vision conversion reports the applicable unverified layout/family status or a matching exact record (see [qualification](#catalog-coverage-and-qualification)).

## Check family and mode coverage

The [shared support registry](../../docs/MODEL_SUPPORT.md) describes implemented
conversion paths, format constraints and Text, Decision and Image eligibility.
Inspect it without downloading a checkpoint:

```sh
uv run --project tools/converter --locked stark-convert --list-families
```

Check source metadata before downloading weights:

```sh
uv run --project tools/converter --locked stark-convert owner/model \
  --preflight --support-mode text
```

Preflight returns JSON with `conditional`, `unsupported` or `inconclusive` status;
all three are report outcomes with exit status 0. It creates no converted model.
A conditional result leaves complete tensor and native checks pending. An
uncatalogued source can qualify for an implemented family path; adapter presence
alone does not establish complete-model support, execution or decision
calibration. `--support-mode` can also require intended-mode eligibility during
actual conversion before weights are requested. The app uses the same path and
pins the checked primary source revision. External catalog metadata is resolved
and checked again during conversion.

## Command reference

```text
stark-convert [SOURCE] [options]
```

| Option | Values | Meaning |
| --- | --- | --- |
| `SOURCE` | repository, URL or directory | Hugging Face repository name or URL, or a local checkpoint directory |
| `--output PATH` | | Output directory; must not exist |
| `--revision REV` | branch, tag or 40-hex commit | Hugging Face revision, resolved to a commit. For a local directory it must be a 40-hex commit |
| `--repo ID` | catalog identity | Treat a local directory as this catalog model; omit with `--metadata-profile hf-generic` |
| `--metadata-profile hf-generic` | | Use the checkpoint's own metadata for a model that is not in the catalog. Cannot be combined with a catalog `--repo` or with `--require-native-support` |
| `--architecture NAME` | `qwen2`, `qwen3`, `qwen3_5`, `llama`, `llamba`, `mistral`, `modernbert`, `gemma2`, `gemma3_text`, `phi3`, `granite`, `smollm3`, `lfm2` | Assert the checkpoint architecture; conversion fails if the checkpoint disagrees. Required with `--capabilities` for a generic probe |
| `--model-kind KIND` | `language` (default), `classifier` | `classifier` is required for ModernBERT. `tts` is accepted by the parser but no adapter implements it |
| `--dtype DTYPE` | `bfloat16` (default), `float16`, `float32` | Output dtype. Only BF16 output can be loaded by the engine |
| `--artifact-mode MODE` | `text` (default), `vision-language` | Full-vision artifacts for the pinned Qwen3.5-0.8B BF16 checkpoint or the supported merged-BF16 [family profile](../../docs/VISION_FAMILY_CONTRACT.md) |
| `--chunk-mib N` | 1 through 1024, default 8 | Live transformation workspace in MiB. Does not bound total process memory |
| `--offline` | | Do not contact Hugging Face; every file must be local or covered by a source manifest |
| `--source-manifest PATH` | | Replay a previously written manifest |
| `--write-source-manifest PATH` | | Write a private manifest describing every selected source file |
| `--require-native-support` | | Require a stored named-build match for the exact output: load-only evidence for text mode, image execution evidence for vision mode. Requires BF16 and a catalog model |
| `--list-families` | | Print the shared family/mode/format matrix without a source or download |
| `--preflight` | | Inspect selected source metadata and print a support report before weight download; creates no output |
| `--support-mode MODE` | `text`, `decision`, `vision-language` | Check intended native use during preflight or conversion; preflight defaults to the artifact mode |
| `--capabilities` | | Report catalog coverage for `SOURCE` without reading or downloading it |
| `--backend` | `auto`, `streaming` | Accepted for compatibility; both select the same implementation |
| `--version` | | Print the converter version |

The command prints one JSON object on success and `{"status":"error","error_category":...}` on failure. See [errors](#errors).

## Catalog coverage and qualification

The declarative [catalog](stark_convert/catalog.json) records 110 reference model identities from the pinned Lalamo inventory plus five Stark extensions. Adapters and metadata profiles implement 90 of them (43 dense language profiles, 40 MLX profiles, two ModernBERT classifier profiles, and the five extensions):

| Adapter | Implemented identities |
| --- | --- |
| Qwen3 | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-4B-Thinking-2507`, `POLARIS-Project/Polaris-4B-Preview` |
| Qwen3 MLX | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}-MLX-{4bit,8bit}`; affine groups 16/32/64/128 |
| Qwen2 | `Qwen/Qwen2.5-Coder-{0.5B,1.5B,3B,7B,14B,32B}-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Qwen3.5 text | Dense `Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B}`; extension `empero-ai/Qwen3.8-4B-Distill`. Declared vision and MTP tensors are omitted; MoE is unsupported |
| Qwen3.5 text MLX | `mlx-community/Qwen3.5-{0.8B,2B,4B,9B}-MLX-{4bit,8bit}`, `mlx-community/Qwen3.5-27B-{4bit,8bit}` |
| LFM2 | Dense `LiquidAI/LFM2-{350M,700M,1.2B,2.6B}` |
| LFM2 MLX | `mlx-community/LFM2-{350M,700M,1.2B,2.6B}-{4bit,8bit}` |
| LLaMBA / Mamba2 | `cartesia-ai/Llamba-1B`, `cartesia-ai/Llamba-1B-4bit-mlx` |
| Llama | `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-{1B,3B}-Instruct`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `Nanbeige/Nanbeige4.1-3B`, `RekaAI/reka-flash-3.1`; extension `unsloth/Llama-3.2-1B-Instruct` |
| Llama MLX | `mlx-community/Llama-3.1-8B-Instruct-4bit`, `mlx-community/Llama-3.2-{1B,3B}-Instruct-{4bit,8bit}` |
| Mistral | `mistralai/Codestral-22B-v0.1`, `mistralai/Devstral-Small-2505`; extensions `mistralai/Mistral-7B-Instruct-{v0.2,v0.3}` |
| Gemma 2 | Extension `google/gemma-2-2b-it` |
| Gemma 3 text | `google/gemma-3-1b-it`, `google/functiongemma-270m-it`, `EssentialAI/rnj-1-instruct` |
| Gemma 3 text MLX | `mlx-community/gemma-3-1b-it-{4bit,8bit}`; multimodal 4B and 27B variants are unsupported |
| Phi | `microsoft/Phi-4-mini-instruct` (tied Phi3 layout with LongRoPE; untied Phi3 is unsupported) |
| Granite | `ibm-granite/granite-{3.1,3.3}-{2b,8b}-instruct` |
| SmolLM3 | `HuggingFaceTB/SmolLM3-3B` |
| SmolLM3 MLX | `mlx-community/SmolLM3-3B-{4bit,8bit}` |
| ModernBERT classifier | `trymirai/chat-moderation-router`, `trymirai/health-related-router` (both repositories currently inaccessible) |

Implementation is tracked per architecture and weight format; qualification is tracked per exact checkpoint, dtype, config, output hash and engine build. In text mode, twenty-seven BF16 catalog artifacts have a stored load-only qualification: when a conversion's source manifest, weights, config and tokenizer hashes match such a record, the receipt reports `native_loading: "qualified_tuple"` together with the named build's worker, source, lockfile and toolchain hashes. That record proves the engine built at that commit loaded the artifact and reached readiness; it does not measure inference quality. Other implemented text identities, size variants, and generic conversions start as `unverified`, or `runtime_blocked` when their dtype or dimensions exceed what the engine loads. `--require-native-support` requires an exact match in the applicable catalog. Remaining unsupported rows fail with an unsupported category.

Vision mode uses a separate [execution catalog](stark_convert/vision_runtime_qualifications.json).
Its records bind all five artifact files and the generation worker to the
tested Metal device class and macOS version. The [execution record](../../docs/archive/runs/2026-10-26-qwen35-image-execution.md)
includes final-logit parity, visual questions, resource observations, and failure
recovery. Conversion returns that named-build evidence when output hashes match;
serving additionally verifies the launched executable and current runtime.
Compatible merged BF16 Qwen3.5 fine-tunes use schema 3 and report
`native_loading: "supported_family_unverified"`. On installation or startup,
the host can enable image input after automatic local checks and labels it
`locally_tested`. Exact catalog matches are `reference_verified`. These local
checks do not create an execution-catalog entry or prove reference parity; see
the [family contract](../../docs/VISION_FAMILY_CONTRACT.md).

Probe coverage without reading a checkpoint:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen2.5-Coder-0.5B-Instruct --capabilities
```

## Future checkpoints and architecture tracking

The `vision-language` artifact mode is tracked separately in the
[native image implementation contract](../../docs/MULTIMODAL_IMPLEMENTATION.md).
It validates and retains the full vision tower, processor and template for the
pinned checkpoint or the merged BF16 dense Qwen3.5 family. New family artifacts
use config-derived version 2: supported decoder/tower dimensions and tensor
shapes come from validated configuration within operation, indexing and storage
bounds. Previously converted fixed version-1 0.8B artifacts remain readable.
Both require recognized tokenizer, template and processor forms. Version 2
additionally accepts a second reviewed tokenizer processing program, the stock
reasoning template and equivalent processor files when their normalized recipes
agree. The two tokenizer programs share vocabulary and special tokens but
intentionally differ for some combining-mark text; conversion preserves the
selected program. Formatting/key-order equivalence cannot admit an arbitrary
new program. Schema 2 and version 1 retain their original tokenizer gate. The
worker, CLI, Local API and Swift app support
one PNG/JPEG image after passing host evidence checks. Default Qwen3.5 conversions
remain text-only. A model's size or repository name does not establish support;
its complete graph and weights must pass admission and its local execution
checks must pass before images are enabled. MoE, LoRA-only repositories,
quantized vision weights, unrecognized operations and new architectures require
additional support. The
[local signed-package replay](../../docs/archive/runs/2026-10-27-qwen35-swift-packaged-app.md)
covers the earlier exact-tuple Swift and GUI checks. Public notarized release
qualification remains pending.

Convert a self-contained merged vision checkpoint with:

```sh
uv run --project tools/converter --locked stark-convert my-org/qwen35-merged \
  --artifact-mode vision-language \
  --write-source-manifest runtime/converter-sources/qwen35-merged.json \
  --output models/qwen35-merged-vision
```

Vision mode selects the family profile automatically from a supported complete
checkpoint. Use a local source path plus `--offline` for a downloaded folder. The converter
uses the checkpoint's own metadata; it does not borrow sidecars from the official
model. Family conversion performs structural validation. Installation checks
execution on the current Mac before registering a schema-3 model. See the
[complete family boundary](../../docs/VISION_FAMILY_CONTRACT.md).

A fine-tune of a supported architecture converts through its adapter without a catalog entry:

```sh
uv run --project tools/converter --locked stark-convert my-org/future-model \
  --metadata-profile hf-generic \
  --write-source-manifest runtime/converter-sources/future-model.json \
  --output models/future-model-stark
```

The adapter is selected from the checkpoint's `model_type` and the full config and tensor layout are validated. `hf-generic` requires self-contained `config.json`, `tokenizer.json` and `tokenizer_config.json`, uses `generation_config.json` and `chat_template.jinja` when present, and requires a supported chat template and explicit EOS identifiers for language models. Duplicate inline and Jinja templates must agree exactly. Generic text conversion uses the checkpoint's own tokenizer, template and generation fields with plain system, user and assistant roles; output parsers, reasoning controls and default system prompts need a reviewed catalog profile. Version-2 vision artifacts receive the reviewed family boolean thinking control. Hugging Face text-only exports with `model_type=qwen3_5_text` select the `qwen3_5` adapter; a nested text or vision wrapper that conflicts with the text layout is rejected.

Trained `Qwen/Qwen3-4B-Instruct-2507` and `empero-ai/Qwen3.8-4B-Distill` checkpoints are covered step by step in the [trained checkpoint guide](../../docs/QWEN_TRAINED_CHECKPOINTS.md).

A self-contained ModernBERT sequence classifier with a declared chat template:

```sh
uv run --project tools/converter --locked stark-convert /absolute/path/to/classifier \
  --metadata-profile hf-generic --model-kind classifier --architecture modernbert \
  --write-source-manifest runtime/converter-sources/future-classifier.json \
  --output models/future-classifier-stark
```

Classifier output uses `ClassifierModelConfig` with numeric label order and the chat token codec, with no generation policy. Only mean pooling, the supported GELU layout and dense weights are accepted; CLS, SEP, PAD, added tokens and post-processor identifiers must agree with the normalised vocabulary. A missing template fails; the converter does not invent one.

Generic conversions record `catalogued=false` and the chosen architecture and profile in the receipt, and a generic source manifest binds that choice for offline replay.

## MLX checkpoints

MLX conversion repacks an existing affine 4-bit or 8-bit checkpoint without requantizing. Packed 32-bit words become U8 tensors with identical payload bytes; scales and biases are cast, fused, transposed and padded to Stark's group and output layout; embedding planes keep their layout. Every compressed matrix needs a complete plane triplet, fused groups must share representation and dtype, and the catalog binds each identity to its declared bit width. Supported MLX families are Qwen3, Qwen3.5 text, Llama, LLaMBA, SmolLM3, Gemma 3 text and LFM2 with groups 16, 32, 64 and 128; quantized Qwen2, Mistral, Granite and Gemma 2 are unsupported.

```sh
uv run --project tools/converter --locked stark-convert mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --output models/qwen35-0.8b-4bit
```

MLX metadata follows each profile's registered source: Qwen3 and Llama use their original repository's tokenizer and generation files, Qwen3.5 uses the official tokenizer and the Qwen3.5-27B generation file, LFM2 uses its own files, SmolLM3 and Gemma use the MLX tokenizer with official generation files, and LLaMBA uses the Llama-3.2-1B files with its registered EOS. External sources are pinned independently; a missing or gated dependency fails rather than substituting metadata. A self-contained future MLX checkpoint can use `hf-generic` with explicit stop metadata and an explicit `quantization` field.

## Experimental Qwen3 GGUF import

A separate experimental route imports Qwen3 Q8_0 GGUF files, keeping the integer codes and rounding the F16 block scales to BF16, then verifies every emitted code and coefficient independently. It needs the original Hugging Face checkpoint directory for metadata:

```sh
uv run --project tools/converter --locked python -m stark_convert.gguf_import \
  --source /path/to/qwen3.gguf --source-revision GGUF_COMMIT \
  --hf-source /path/to/Qwen3-0.6B --hf-revision HF_COMMIT \
  --output models/qwen3-q8
```

The Hugging Face directory must include `generation_config.json`. Only Qwen3 Q8_0 is supported; other quantization types fail. On failure the command prints `{"status":"failed","reason":"unsupported_or_invalid_gguf_experiment"}` for every cause, including an existing output directory. Its receipt records absolute local source paths. The [GGUF experiment](../../docs/benchmarks/QWEN3_GGUF_IMPORT_EXPERIMENT.md) and [prefill optimization](../../docs/benchmarks/QWEN3_GGUF_PREFILL_OPTIMIZATION.md) studies record its correctness and performance checks.

## Offline replay and source manifests

`--write-source-manifest` writes an owner-only file listing every selected source file with its path, revision, size, hash and signature, including cross-repository metadata sources. It is private operational data; keep it under ignored storage. Replay it without network access:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B \
  --source-manifest runtime/converter-sources/qwen3-06b.json --offline \
  --output models/qwen3-06b-replayed
```

Replay checks required files, immutable external revisions, paths, content hashes and source identity before conversion, and re-hashes every consumed input before publication. File signatures are checked before and after each hash batch, and changed, missing or incomplete inputs fail. Manifests record cache paths, so a copied manifest needs the referenced files to remain available.

## Memory and limits

`--chunk-mib` bounds the live transformation workspace (default 8 MiB). Header JSON, tensor catalogs, tokenizer normalisation, hashing and the interpreter use memory outside that bound; the benchmark tool reports sampled process RSS separately. Metadata planning has fixed limits of 4,096 layers, 262,144 source tensors and 32 tensor dimensions, and the declared layer count cannot exceed the tensor inventory. Safetensors byte products are checked against the validated byte span.

Source verification reads every consumed byte twice in addition to the conversion read (once when resolving the source, once before publication), which is the main reason dense conversion of large checkpoints is slower than converters that skip it. Hashing of independent large shards uses up to four workers.

Metadata sidecars must resolve inside the selected checkpoint, or inside a canonical Hugging Face cache's sibling `blobs` directory; other links and special files are rejected. The main route opens inputs without following symlinks; the GGUF route does not apply that check.

## Benchmark tool

```sh
uv run --project tools/converter --locked stark-convert-benchmark \
  --source /absolute/path/to/checkpoint --repo Qwen/Qwen3-0.6B \
  --source-manifest runtime/converter-sources/qwen3-06b.json \
  --reference-output /absolute/path/to/frozen-reference-output \
  --output runtime/converter-benchmark/run --repeats 3
```

Each repeat runs a fresh converter process after warming the source files; setup and downloads are excluded. `--reference-output` compares tensor names, shapes, dtypes and payload hashes, tensor specs, config and normalised tokenizer against a frozen artifact; whole-file hashes can differ when serialization order differs. For an uncatalogued folder, omit `--repo` and add `--metadata-profile hf-generic`; `--architecture` and `--model-kind` are forwarded. Reports are written atomically with private permissions; an interrupted run keeps completed measurements and marks the failed repeat. Recorded results are collected in [conversion benchmarks](../../docs/benchmarks/CONVERSION_BENCHMARKS.md).

Private metadata parity tests read supplied reference artifacts from `STARK_CONVERT_REFERENCE_ARTIFACTS` (`<root>/<repo with slash replaced by __>/provenance.json`, `sources/<repo>/...`, `expected/config.json`, `expected/tokenizer.json`); they never install Lalamo.

## Install outside the checkout

Build the wheel, export the locked dependencies and install into a managed environment:

```sh
uv build tools/converter --python 3.13.13 --out-dir runtime/converter-package
uv export --project tools/converter --locked --no-dev --no-emit-project \
  --output-file runtime/converter-package/requirements.txt
uv venv --python 3.13.13 /absolute/path/to/converter-env
uv pip install --python /absolute/path/to/converter-env/bin/python --require-hashes \
  -r runtime/converter-package/requirements.txt
uv pip install --python /absolute/path/to/converter-env/bin/python --no-deps \
  runtime/converter-package/stark_convert-0.9.1-py3-none-any.whl
/absolute/path/to/converter-env/bin/stark-convert --version
```

Installed conversions use the packaged provenance inputs. The package is not published to an index.

## Errors

Failures print `{"status":"error","error_category":CATEGORY}` to standard error and exit with status 1 (130 when interrupted). `CATEGORY` is `invalid_arguments` for argument errors, `interrupted` for Ctrl-C, and otherwise the Python exception class name, for example `UnsupportedConversion`, `ConversionError`, `GatedRepoError` or `FileNotFoundError`. Exception text, paths and native diagnostics are never printed. Common causes:

| Category | Typical cause |
| --- | --- |
| `invalid_arguments` | Unknown option, missing `--output`, or a value outside its range |
| `UnsupportedConversion` | Architecture, layout, dtype, quantization or metadata the converter does not implement |
| `ConversionError` | Source changed during conversion, failed verification, or an existing output directory |
| `ValueError` with no further detail | Internal validation, including `unaccounted_package_resource`: a file with a suffix other than `.py`, `.json`, `.toml`, `.txt` or `.md` (such as `.DS_Store`) inside the installed `stark_convert` package makes every conversion fail until it is removed |
| `GatedRepoError`, `RepositoryNotFoundError`, `LocalEntryNotFoundError` | Hugging Face access or offline-mode problems |

## Provenance and policy

Lalamo at [`831accb8252a6b6606c378048caeb44178f1a050`](https://github.com/trymirai/lalamo/tree/831accb8252a6b6606c378048caeb44178f1a050) is the pinned design and qualification reference; its [MIT license](LICENSE.lalamo) and Mirai Tech Inc. copyright are preserved. The catalog exporter under `scripts/` reads that source in a separate developer environment; the runtime reads only the checked-in catalog and uses Stark's own adapters, with no Lalamo, JAX or PyTorch dependency. Adapted source areas are the model specs, origins, Hugging Face loaders and per-architecture configs, the normalization, linear, embedding, RoPE, DeltaNet and short-convolution contracts, `weight_matrix.py`, the MLX and quantization-parameter handling, packing utilities and safetensors reading.

Converter receipts hash the conversion-affecting package resources and the exporter and record the dependency lock separately. Model files, private source manifests and native validation artifacts stay out of Git; only aggregate results and hashes are published. The milestone records that established each adapter are under [docs/archive/conversion](../../docs/archive/conversion/).
