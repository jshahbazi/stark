# Stark model converter

Convert supported Hugging Face models into Stark's `config.json`, `tokenizer.json`, and `model.safetensors` files.

## Set up

Install `uv`, then run these commands from the repository root. The converter uses Python 3.13.13.

```sh
uv sync --project tools/converter --locked
```

## Convert from Hugging Face

Pass the model's Hugging Face repository name:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B \
  --output models/qwen3
```

This downloads the model and writes the converted files to `models/qwen3`. Replace the repository name and output path for another [supported model](#catalog-coverage-and-qualification). Gated models require access through your existing Hugging Face credentials.

## Convert a downloaded folder

Point the converter at your local model folder:

```sh
uv run --project tools/converter --locked stark-convert /path/to/downloaded-model \
  --metadata-profile hf-generic --offline --output models/local-model
```

Replace `/path/to/downloaded-model` with your folder's path. This conversion uses local model files and needs no Hugging Face account.

Keep the full safetensors checkpoint together. The folder needs:

- `config.json`
- `tokenizer.json` and `tokenizer_config.json`
- All `.safetensors` files, plus the index file if the weights are split into shards

Language models also need a chat template and end-of-sequence (EOS) token IDs in their metadata. The converter reads `generation_config.json` and `chat_template.jinja` when present. See [metadata requirements](#future-checkpoints-and-architecture-tracking) for details.

## Find the result

Your output folder contains the converted config, tokenizer and weights, plus a `conversion.json` receipt. Use a new output folder each time; existing folders are refused.

BF16 is the default output format. Conversion validates a supported model layout. Engine loading is tracked separately; generic conversions start native-unverified.

## Optional settings

| Need | Option |
| --- | --- |
| Convert a specific Hugging Face revision | `--revision COMMIT` |
| Save source information for offline replay | `--write-source-manifest PATH` |
| Require an exact recorded engine loading check | `--require-native-support` |

For a downloaded model in the [catalog](#catalog-coverage-and-qualification), use its registered profile:

```sh
uv run --project tools/converter --locked stark-convert /path/to/downloaded-model \
  --repo Qwen/Qwen3-0.6B --offline --output models/local-qwen3
```

Some catalog profiles use metadata from another repository. Those profiles need a complete source manifest for offline conversion. Use the generic command above when your model folder has all its own metadata and you want the generic profile's plain defaults.

See [standalone installation](#install-outside-the-checkout) for using a wheel outside this checkout. The [hardening checklist](../../docs/MODEL_CONVERSION_HARDENING.md) tracks current tool work; further model coverage is deferred.

## Experimental Qwen3 GGUF import

The separate experimental `python -m stark_convert.gguf_import` route imports Qwen3 Q8_0 GGUF files with original HF metadata sidecars. It preserves integer compression and rounds F16 scales to native BF16, then independently verifies every emitted code and coefficient. The [Qwen3 GGUF experiment](../../docs/QWEN3_GGUF_IMPORT_EXPERIMENT.md) records scoped correctness, memory, and performance comparisons. It does not promote catalog qualification or add other GGUF quantization types.

## Catalog coverage and qualification

The attributed [catalog](stark_convert/catalog.json) accounts for **110 built-in reference entries**, with **four explicitly separated Stark extensions**. Implemented adapters and metadata profiles cover **85 built-ins plus four extensions**, or 89 identities. This includes 43 dense language profiles, 40 MLX profiles and two dense ModernBERT classifier profiles:

| Adapter | Implemented identities |
| --- | --- |
| Qwen3 | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-4B-Thinking-2507`, `POLARIS-Project/Polaris-4B-Preview` |
| Qwen3 MLX | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}-MLX-{4bit,8bit}`; affine groups 16/32/64/128, divisible logical input widths |
| Qwen2 | `Qwen/Qwen2.5-Coder-{0.5B,1.5B,3B,7B,14B,32B}-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Qwen3.5 text | Dense `Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B}`; declared vision/MTP tensors omitted for text conversion; MoE remains unsupported |
| Qwen3.5 text MLX | `mlx-community/Qwen3.5-{0.8B,2B,4B,9B}-MLX-{4bit,8bit}` and `mlx-community/Qwen3.5-27B-{4bit,8bit}` |
| LFM2 | Dense `LiquidAI/LFM2-{350M,700M,1.2B,2.6B}`; short-convolution and attention layers |
| LFM2 MLX | `mlx-community/LFM2-{350M,700M,1.2B,2.6B}-{4bit,8bit}` |
| LLaMBA / Mamba2 | `cartesia-ai/Llamba-1B` and `cartesia-ai/Llamba-1B-4bit-mlx`; backbone and alternating SSM/MLP layouts |
| Llama | `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.2-{1B,3B}-Instruct`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `Nanbeige/Nanbeige4.1-3B`, `RekaAI/reka-flash-3.1`; extension `unsloth/Llama-3.2-1B-Instruct` |
| Llama MLX | `mlx-community/Llama-3.1-8B-Instruct-4bit`, `mlx-community/Llama-3.2-{1B,3B}-Instruct-{4bit,8bit}` |
| Mistral | `mistralai/Codestral-22B-v0.1`, `mistralai/Devstral-Small-2505`; extensions `mistralai/Mistral-7B-Instruct-{v0.2,v0.3}` |
| Gemma 2 | Extension `google/gemma-2-2b-it`, using the pinned Gemma2 config and standard LanguageModelSpec |
| Gemma 3 text | `google/gemma-3-1b-it`, `google/functiongemma-270m-it`, `EssentialAI/rnj-1-instruct` |
| Gemma 3 text MLX | `mlx-community/gemma-3-1b-it-{4bit,8bit}`; multimodal 4B/27B variants remain unsupported |
| Phi | `microsoft/Phi-4-mini-instruct`, the pinned tied Phi3 layout with LongRoPE; untied Phi3 is unsupported |
| Granite | `ibm-granite/granite-{3.1,3.3}-{2b,8b}-instruct` |
| SmolLM3 | `HuggingFaceTB/SmolLM3-3B` |
| SmolLM3 MLX | `mlx-community/SmolLM3-3B-{4bit,8bit}` |
| ModernBERT classifier | `trymirai/chat-moderation-router`, `trymirai/health-related-router`; mean pooling and GELU sequence-classification layout; both repositories currently inaccessible and unqualified |

This is implementation coverage. Exact checkpoint/dtype/config qualification remains per tuple and is recorded in the milestone evidence. A passing small checkpoint does not qualify every size or config variant in its family. Other catalog entries, compressed schemes, custom architectures/RoPE variants, and unsupported metadata representations fail with a sanitized unsupported category. Future repositories can explicitly use the generic path below when their architecture, tensor layout and metadata pass the same validation.

The preserved dense milestone 4 adds `Qwen/Qwen3.5-0.8B` and `LiquidAI/LFM2-350M`, with exact complete reference parity. Qwen3.5 conversion extracts the text decoder from its declared wrapper, preserving source verification of omitted vision/MTP tensors. It preserves gated attention, DeltaNet projection ordering and F32 recurrent/convolution/normalization parameters. LFM2 retains attention/short-convolution layer order, accepted 2D/3D convolution shapes and its feed-forward dimension rules. Qwen3.5 MoE, later Qwen/LFM generations, multimodal execution and scoring/generation quality remain unqualified or unsupported.

The [MLX continuation](../../docs/MODEL_CONVERSION_MILESTONE_4_MLX.md) adds exact group-64 BF16 qualifications for `mlx-community/Qwen3.5-0.8B-MLX-{4bit,8bit}` and `mlx-community/LFM2-350M-{4bit,8bit}`. All four match every pinned reference tensor/spec/config/tokenizer and pass named-build native readiness/EOF with zero scoring calls. [Milestone 5](../../docs/MODEL_CONVERSION_MILESTONE_5.md) adds dense and group-128 4-bit LLaMBA BF16 qualifications. There are now twenty-seven exact qualified tuples; all other implemented identities and variants retain their separate unqualified status.

## Future checkpoints and architecture tracking

Tensor conversion is implemented by **architecture and weight format**. Checkpoint records separately track metadata routing, exact source revisions, output hashes and native qualification. A new fine-tune of a supported architecture can reuse its adapter without adding a catalog entry:

```sh
uv run --project tools/converter --locked stark-convert my-org/future-model \
  --metadata-profile hf-generic \
  --write-source-manifest runtime/converter-sources/future-model.json \
  --output models/future-model-stark
```

The converter selects the adapter from the checkpoint's `model_type` and validates the full config and tensor layout. Optional `--architecture llama` asserts the expected type; it cannot override a conflicting checkpoint. Accepted architecture identifiers are `qwen2`, `qwen3`, `qwen3_5`, `llama`, `llamba`, `mistral`, `modernbert`, `gemma2`, `gemma3_text`, `phi3`, `granite`, `smollm3`, and `lfm2`. ModernBERT requires `--model-kind classifier`; the other adapters require the default language kind. Dense BF16/F16/F32 uses the corresponding strict planner. Existing affine MLX 4/8-bit support applies to Qwen3, Qwen3.5 text, Llama, LLaMBA, SmolLM3, Gemma3 text and LFM2; the pinned LLaMBA catalog has one 4-bit profile. A different architecture or unsupported variant still needs implementation.

`hf-generic` requires self-contained `config.json`, `tokenizer.json`, and `tokenizer_config.json`; it selects `generation_config.json` and `chat_template.jinja` whenever present. Language models require a supported chat template and explicit nonempty EOS identifiers in the model or generation metadata. Duplicate inline/Jinja templates must agree exactly. It uses the checkpoint's tokenizer, template and supported generation fields, with plain system/user/assistant roles. Output parsers, reasoning controls and system-prompt defaults require a separately reviewed checkpoint profile. Registered profiles retain their authoritative metadata and packing rules and refuse generic overrides.

For a self-contained ModernBERT sequence classifier with a declared chat template:

```sh
uv run --project tools/converter --locked stark-convert /absolute/path/to/classifier \
  --metadata-profile hf-generic --model-kind classifier --architecture modernbert \
  --write-source-manifest runtime/converter-sources/future-classifier.json \
  --output models/future-classifier-stark
```

Classifier output uses `ClassifierModelConfig`, numeric label order and its chat token codec, with no generation policy. BOS/EOS codec fields are optional; CLS/SEP/PAD, added tokens, tokenizer padding and post-processor identifiers must agree with the normalized vocabulary. Only mean pooling, the supported GELU layout and dense weights are accepted. `decoder_bias` is an unused masked-LM annotation for this architecture; unexpected masked-LM tensors still fail. A present generic generation sidecar remains source-verified but is not interpreted. Missing templates fail explicitly; the converter does not invent one. Anonymous local checkpoints and schema-2 offline replay retain the same model-kind checks.

Generic conversions emit `catalogued=false` and the chosen architecture/profile in their receipt. New checkpoints remain native-unverified, or runtime-blocked when their requested dtype/config exceeds current native limits. `--require-native-support` refuses them until an exact registered qualification exists. A generic private schema-2 source manifest binds the architecture/profile and optional-sidecar presence; offline replay restores that mode and detects source changes. Existing catalog schema-1 replay identities remain unchanged.

For a source-free probe of a future checkpoint, declare both `--metadata-profile hf-generic` and `--architecture`; the probe cannot infer a checkpoint architecture without reading it. Probe results do not confer qualification. See the [tracking and qualification boundaries](../../docs/MODEL_CONVERSION_MILESTONE_4.md).

MLX conversion repacks existing affine checkpoints without computing new quantization. Little-endian packed I32/U32 words become U8 tensors with identical payload bytes. Projection scales and affine biases are cast, fused, transposed, and padded to Stark's group/output layout; input and tied embedding planes retain their input/output layout. A complete plane triplet is required for every compressed matrix. Fused groups must share representation and coefficient dtype. The catalog binds each compressed identity to its declared 4-bit or 8-bit format, and contradictory declarations or layouts fail.

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B-MLX-4bit \
  --revision 173234aa840d113125e9f2271100ddbaf16c9620 \
  --write-source-manifest runtime/converter-sources/qwen3-06b-mlx4.json \
  --output models/qwen3-06b-mlx4-stark
```

All MLX families use affine groups 16/32/64/128 and the same bounded coefficient operations. Llama preserves its accepted Llama 3 RoPE; SmolLM3 preserves its per-layer RoPE mask, omitted tied-head default and accepted projection biases. Gemma text retains runtime embedding scaling, normalization offsets and local/global attention/RoPE semantics. Qwen3.5 applies every gated-attention/DeltaNet row permutation to all three source planes, and preserves MLX's baked normalization weights. Its primary `quantization` field is required; alias-only `quantization_config` is rejected. LFM2 selects convolution layout from the actual embedding planes, accepting kernel-first flat and both 3D source forms; flat sources use bounded transpose-and-F32 widening. Unfused projections can remain dense, while mixed representations within a fused projection fail. Quantized Qwen2, Mistral, Granite and Gemma2 retain their unsupported status.

MLX metadata follows each registered source rule: Qwen3 uses its original repository's tokenizer/generation files; Qwen3.5 uses its corresponding official tokenizer and the Qwen3.5-27B generation file; LFM2 uses its own tokenizer/Jinja template and reviewed static generation defaults. Llama uses the MLX tokenizer and official generation file; SmolLM3 uses its MLX tokenizer and original chat-template/generation files; Gemma text uses its MLX tokenizer and official generation file. External sources are independently pinned. Missing or gated dependencies fail rather than silently replacing metadata. An explicit private offline manifest can reference verified local content; the separate qualification evidence records any public mirror transport and its exact canonical Git-blob match.

LLaMBA uses the official Llama-3.2-1B tokenizer/generation files and its registered fixed EOS property. Its config-less MLX repository selects the dense LLaMBA config externally; profile bits plus validated matrix planes establish group size. Unselected primary config files never override that source. The real MLX profile has a dense tied embedding and an unused approximate `head.linear`; all three head planes are validated, omitted under the declared tied semantics and retained in full source verification. Other head/layout variants fail or require exact duplicate-plane comparison. A self-contained future `model_type=llamba` checkpoint can use `hf-generic` with its own explicit stop metadata and, for MLX, explicit quantization.

```sh
uv run --project tools/converter --locked stark-convert mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --output ./models/qwen35-0.8b-4bit
uv run --project tools/converter --locked stark-convert mlx-community/LFM2-350M-8bit \
  --output ./models/lfm2-350m-8bit
```

The Qwen3-0.6B group-128 BF16 tuples and selected Llama-3.2-1B, SmolLM3-3B and Gemma3-1B group-64 BF16 tuples have exact artifact parity and named-build load-only qualifications at both bit widths. Other sizes and groups remain unqualified. No compressed inference-speed or quality claim is made.

Probe the catalog without reading or downloading a checkpoint:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --capabilities
```

The probe reports catalog implementation, model kind, and runtime status with `source_validation=not_performed`. Successful conversion validates the complete accepted source layout. Implemented adapters emit language-model or ModernBERT classifier root configs. TTS, Mamba1 and other unimplemented state-space families, MoE, and other compressed conversion remain later workstreams.

Stark's supported dense loader selects BF16 weights. F16/F32 outputs are conversion-only and are marked runtime-blocked. A catalog probe does not inspect source or output artifacts. BF16 conversions retain their planner's status: earlier families report `supported_layout_unverified_tuple`; Qwen3.5, LFM2, LLaMBA, ModernBERT and generic routes remain `unverified`, with `runtime_blocked` for unsupported native dimensions/configs. An exact match of registered identity, source-manifest hash, weights hash, config hash, and tokenizer hash promotes the result to `qualified_tuple` using the stored named-build record.

`--require-native-support` requires that exact stored match before publication; unmatched artifacts fail cleanly. A match reports `native_loading=qualified_tuple` and emits the full `native_qualification` record, including the named build's worker/source/Cargo-lock/toolchain hashes, Rust 1.98.1, release profile, features, and load-only result. That result applies to the recorded build and exact artifacts. It provides no guarantee for another current or future engine binary. Execution remains unverified: readiness and EOF do not establish scoring, generation, classification, or TTS correctness. Stark currently rejects MoE construction.

## Measured improvements over Lalamo

The preserved version 0.9.0 benchmarks two public real-weight ModernBERT fixtures with explicit local synthetic chat-template overlays and BF16 output:

| Public classifier fixture | Stark median | Lalamo median | Lower time | Lower sampled RSS |
| --- | ---: | ---: | ---: | ---: |
| MoritzLaurer BF16 source | 0.562 s | 1.137 s | 50.59% | 84.90% |
| SoFairOA F32 source | 0.861 s | 1.312 s | 34.36% | 89.58% |

All twenty serial fresh warm-file commands pass full tensor/spec/config/tokenizer parity. All ten candidate outputs match their frozen private exact source/artifact/load-only proofs, while generic receipts remain native-unverified and have no public ledger qualification. The F32 timing ranges overlap; five repeats on one host establish these measurements only. Templates are explicit fixture additions; no direct unchanged public-repository conversion or inaccessible Mirai qualification is claimed. Full source verification remains enabled. [Milestone 6 evidence](../../docs/MODEL_CONVERSION_MILESTONE_6.md) and [independent review](../../docs/MODEL_CONVERSION_MILESTONE_6_REVIEW.md) record scopes, ranges, hashes and cleanup. Classification correctness, quality and inference speed remain unverified.

The preserved version 0.8.0 adds dense and 4-bit MLX LLaMBA benchmarks. Five serial warm-file complete commands per backend/tuple measured:

| LLaMBA checkpoint, BF16 output | Stark median | Lalamo median | Lower sampled RSS |
| --- | ---: | ---: | ---: |
| 1B 4-bit MLX, group 128 | 2.142 s | 2.225 s | 96.42% |
| 1B, F32 source | 5.791 s | 2.105 s | 97.13% |

All twenty outputs pass exact tensor/spec/config/tokenizer parity, and all ten candidate commands match exact named-build load-only qualifications. MLX's median is 3.73% lower, with overlapping ranges. Dense conversion is 2.75 times as slow and its latency objective remains open. Candidate timing includes full initial/final source verification; Lalamo's source/receipt checks are weaker. These descriptive one-host results make no inference-speed or quality claim. [Milestone 5 evidence](../../docs/MODEL_CONVERSION_MILESTONE_5.md) and [independent review](../../docs/MODEL_CONVERSION_MILESTONE_5_REVIEW.md) retain all observations, hashes and limits.

The preserved version 0.7.0 adds four hybrid MLX benchmarks with exact parity, named-build loading and the same 8 MiB workspace. Five serial warm-file complete commands per backend/tuple measured:

| MLX checkpoint, group 64 BF16 | Stark median | Lalamo median | Lower time | Lower sampled RSS |
| --- | ---: | ---: | ---: | ---: |
| LFM2-350M 4-bit | 0.532 s | 1.853 s | 71.3% | 92.0% |
| LFM2-350M 8-bit | 0.623 s | 1.581 s | 60.6% | 92.6% |
| Qwen3.5-0.8B 4-bit text | 0.997 s | 2.059 s | 51.6% | 94.9% |
| Qwen3.5-0.8B 8-bit text | 1.393 s | 1.940 s | 28.2% | 95.1% |

All forty outputs pass exact parity, and all twenty candidate commands match their recorded native qualification. Initial/final source verification stays enabled. Observed ranges are separated in this run; the five repeats on one host establish these measurements only. No inference speed or quality claim follows. The [new evidence](../../docs/MODEL_CONVERSION_MILESTONE_4_MLX.md) records full ranges, frozen hashes, protocol, memory-sampling limits and [independent review](../../docs/MODEL_CONVERSION_MILESTONE_4_MLX_REVIEW.md).

The preserved version 0.6.0 adds two dense hybrid architecture benchmarks with exact parity, named-build loading and the same 8 MiB workspace. Five serial warm-file complete commands per backend/tuple measured:

| Dense BF16 output | Stark median | Lalamo median | Median sampled RSS reduction |
| --- | ---: | ---: | ---: |
| LFM2-350M | 1.233 s | 1.688 s | 90.2% |
| Qwen3.5-0.8B text | 3.525 s | 2.707 s | 91.4% |

LFM2's median time is 27% lower. Qwen3.5's is 30% higher, so its latency objective remains open. Both observed timing ranges are broad and overlap; all twenty observations are preserved. Candidate commands include complete source verification and exact native qualification checks. These measurements grant no inference speed/quality claim. Full ranges, source/build hashes, protocol and [independent review](../../docs/MODEL_CONVERSION_MILESTONE_4_REVIEW.md) are in the [milestone 4 evidence](../../docs/MODEL_CONVERSION_MILESTONE_4.md).

The preserved version 0.5.1 packet exports MLX coefficients sequentially, preserving inline output hashing and avoiding a completed-model reread. Complete median conversion time fell **16–38%** against the frozen prior implementation for six group-64 BF16 tuples. Five measured medians beat Lalamo, and median sampled process-tree RSS is **90.7–96.6% lower**. The writer copies packed codes directly and keeps transformation workspace at most 8 MiB. The runtime has no Lalamo, JAX or PyTorch dependency. Full initial and final source/content verification remains enabled.

| MLX checkpoint | Stark complete median | Lalamo complete median | Sampled RSS reduction |
| --- | ---: | ---: | ---: |
| Llama3.2-1B 4-bit | 1.256 s | 2.397 s | 94.2% |
| Llama3.2-1B 8-bit | 2.202 s | 2.604 s | 96.6% |
| SmolLM3-3B 4-bit | 2.779 s | 3.377 s | 95.4% |
| SmolLM3-3B 8-bit | 5.083 s | 4.233 s | 96.2% |
| Gemma3-1B 4-bit | 1.551 s | 2.580 s | 90.7% |
| Gemma3-1B 8-bit | 2.464 s | 2.537 s | 90.8% |

Llama 4-bit and Gemma 4-bit measured 1.91× and 1.66× complete-command ratios. SmolLM3 8-bit remains 20% slower than Lalamo, despite its 28% improvement against the previous branch. Gemma 8-bit's advantage is about 3%; its observed ranges overlap, as do Llama 8-bit's. Broad latency parity remains open. Earlier Qwen3-0.6B and dense-family measurements retain their separate source states. Ratios describe conversion and grant no inference or quality qualification.

Each of the frozen baseline, final candidate and external Lalamo ran five serial fresh processes per tuple with warm selected files and rotating/reversed order: 90 complete commands. All passed exact tensor/spec/config/tokenizer parity. Every baseline and candidate also matched the previous whole-file artifact hashes and exact native qualification. Those unchanged bytes retain the prior named-build load-only scope; this packet made zero scoring or generation calls. Setup/download, warming and comparisons were excluded. Complete commands include verification, metadata, writing, hashes and receipts; Lalamo performs weaker source/receipt verification. RSS was sampled every 10 ms and is separate from the workspace limit. Exact sources, timing ranges, host versions, failed hypotheses and [independent review](../../docs/MODEL_CONVERSION_MILESTONE_3_LATENCY_REVIEW.md) are in the [current evidence](../../docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md).

## Offline replay and source manifests

`--write-source-manifest` creates an owner-only private replay file containing every selected source path, revision, size, hash, and signature. It preserves explicit cross-repository metadata routing, including Devstral's tokenizer, generation, and JSON-field chat-template sources. The manifest file is private operational data and stays under ignored storage.

Replay an existing complete manifest with network access disabled:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B \
  --source-manifest runtime/converter-sources/qwen3-06b.json --offline \
  --output models/qwen3-06b-replayed
```

The replay validates required files, immutable external revisions, paths, content hashes, and source identity before conversion, then fully rehashes consumed inputs before publication. Independent large shards use at most four hashing workers, with reusable digest buffers totaling at most 1 MiB on pinned CPython 3.13.13. Every file signature is checked before and after the complete hash batch, root/index selection is rechecked, and all workers are joined on errors or interruption. Missing files, changed files, and incomplete external dependencies fail. A local primary source uses content hashes when no remote revision exists. Source manifests retain cache paths, so copied manifests require those referenced local files to remain available.

## Memory, validation, and measurements

The writer plans names, shapes, dtype/spec metadata, input consumption, and byte offsets from checkpoint headers before writing. Shared typed operations copy/cast dense tensors, fuse projection rows across shards, transpose bounded tiles when needed, and widen normalization scales to F32. Phi gate/up rows are reordered through contiguous bounded views. Granite residual/logit scaling preserves source-precision arithmetic before target casts, including pinned JAX CPU reciprocal and underflow semantics. Gemma normalization, attention softcaps, and local/global RoPE are explicit configs; SmolLM3 preserves its per-layer RoPE mask. Tied output heads are checked before omission. Synthetic cast/scalar fixtures cover rounding boundaries, subnormals, signed zero, infinities, and NaN payloads against independently generated reference hashes, including every BF16/F16 bit pattern for scalar transforms.

`--chunk-mib` sets the enforced live transformation workspace, defaulting to 8 MiB. Header JSON, tensor catalogs, tokenizer normalization, hashing, and interpreter costs contribute additional memory. Sampled process RSS is reported separately; this option does not impose a total-process memory ceiling.

```sh
uv run --project tools/converter --locked pytest tools/converter/tests -q
uv run --project tools/converter --locked ruff check tools/converter
uv run --project tools/converter --locked ruff format --check tools/converter
uv run --project tools/converter --locked stark-convert-benchmark \
  --source /absolute/path/to/checkpoint --repo Qwen/Qwen3-0.6B \
  --source-manifest runtime/converter-sources/qwen3-06b.json \
  --reference-output /absolute/path/to/frozen-reference-output \
  --output runtime/converter-benchmark/run --repeats 3
```

The benchmark starts a fresh streaming process for every repeat, warms source files, and excludes setup/downloads. `--reference-output` is optional and compares independently generated frozen artifacts: tensor names/shapes/dtypes/payload hashes, tensor specs/logical metadata, complete config, and normalized tokenizer. Reference generation occurs externally before timed runs. The benchmark does not measure reference conversion time or produce a speedup ratio. Whole-file safetensors hashes may differ when serialization order differs while all tensor payloads match.

Version 0.9.1 uses the current installed Python for fresh processes and records that launcher in schema-3 reports. Historical measurements retain their original protocols. To benchmark an anonymous local folder, omit `--repo` and add `--metadata-profile hf-generic`; `--architecture` and `--model-kind` are forwarded to conversion. Reports are saved atomically with private permissions throughout the run. Errors and interruption preserve completed measurements and identify the failed repeat; an incomplete or failed report never qualifies as passing evidence. See the [failure contract](../../docs/MODEL_CONVERSION_HARDENING.md#benchmark-report-contract).

Reported complete CLI time includes startup, conversion, verification, and receipt hashing. Process-tree RSS is sampled every 10 ms; shared pages can be counted more than once and brief peaks may be missed. The harness rejects source, reference, and converter-input changes during the run. Measurements run serially with other model-dependent work paused.

The optional metadata parity tests consume supplied private artifacts through `STARK_CONVERT_REFERENCE_ARTIFACTS`; they never install or invoke Lalamo. Each artifact profile uses `<root>/<repo with slash replaced by __>/provenance.json`, `sources/<repo>/...`, and `expected/config.json` plus `expected/tokenizer.json`. The provenance field `reference_revision` must equal the pin below. Unsupplied profiles remain unqualified.

Models and permitted native aggregate/hash validation artifacts remain in ignored private storage. Only load-only readiness/EOF aggregates and hashes were retained. Publish aggregate outcomes and hashes only. Do not capture prompts, responses, token vectors, decision probabilities, credentials, or exception dumps. The [historical prototype evidence](../../docs/MODEL_CONVERSION.md) retains its original implementation and lock hashes; its timing ratios apply to that prototype only.

## Install outside the checkout

Version 0.9.1 supports local wheel installation. Build and export locked dependencies from the repository, then install into a managed Python environment:

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
/absolute/path/to/converter-env/bin/stark-convert /absolute/path/to/downloaded-model \
  --metadata-profile hf-generic --offline --output /absolute/path/to/converted-model
```

Installed conversion and benchmarking use packaged provenance inputs. Editable installs remain checkout-bound. The bundled lock identifies intended dependencies; the hash-checked export enforces their versions. Package publication remains separate work.

## Conversion behavior

The converter uses independent streaming adapters. It has no Lalamo, JAX or PyTorch runtime dependency; unsupported combinations fail explicitly.

`--backend auto` and `--backend streaming` both use the independent implementation. BF16 is the default output dtype. All implemented dense adapters perform bounded BF16/F16/F32 conversion with frozen reference-byte tests. Mixed source dtypes within a fused group fail explicitly. Existing output directories are refused. The converter stages with owner-only permissions, validates complete supported layouts, and publishes using an atomic no-replace rename on macOS or Linux. Failed conversions remove staging output.

The command emits aggregate timings, hashes, and capability status. Its private `conversion.json` receipt binds source revisions/content hashes, converter inputs, the dependency lock, emitted files, and transformation workspace use. Errors expose fixed categories without exception text or native diagnostics.

## Validation history

The [replacement plan](../../docs/MODEL_CONVERSION_PLAN.md) targets the complete built-in inventory at the pinned Lalamo reference revision. [Milestone 6](../../docs/MODEL_CONVERSION_MILESTONE_6.md) adds ModernBERT classifier roots and generic classifier conversion; its public validation fixtures are distinct from the two inaccessible catalog routers. [Milestone 5](../../docs/MODEL_CONVERSION_MILESTONE_5.md) retains its dense and 4-bit MLX LLaMBA qualifications and open dense latency objective. [Milestone 4](../../docs/MODEL_CONVERSION_MILESTONE_4.md) adds dense Qwen3.5 text/DeltaNet, LFM2 short-convolution/attention and explicit conversion of future checkpoints through supported architecture adapters. Its [MLX continuation](../../docs/MODEL_CONVERSION_MILESTONE_4_MLX.md) adds existing affine 4/8-bit weights for both hybrid architectures. [Milestone 3 latency evidence](../../docs/MODEL_CONVERSION_MILESTONE_3_LATENCY.md) records the sequential MLX writer and its fresh comparison against the previous implementation and Lalamo. The [dense MLX qualification packet](../../docs/MODEL_CONVERSION_MILESTONE_3_DENSE.md), [first MLX packet](../../docs/MODEL_CONVERSION_MILESTONE_3.md), [milestone 2](../../docs/MODEL_CONVERSION_MILESTONE_2.md) and [milestone 1](../../docs/MODEL_CONVERSION_MILESTONE_1.md) retain their original source versions and results.

## Provenance and replacement policy

Lalamo serves as a pinned design and qualification reference at [`831accb8252a6b6606c378048caeb44178f1a050`](https://github.com/trymirai/lalamo/tree/831accb8252a6b6606c378048caeb44178f1a050). The catalog exporter reads that source in a separate developer environment. Runtime conversion reads the checked-in declarative catalog and uses Stark implementations. [Lalamo's MIT license](LICENSE.lalamo) and Mirai Tech Inc. copyright are preserved. Stark retains Uzu base revision `8d01581474760bd1fc7f4a547ff1d02f7dada437`.

Adapted source paths include `model_import/model_spec.py`, `model_import/model_specs/`, `model_import/origins.py`, `model_import/loaders/huggingface.py`, `model_import/model_configs/huggingface/{qwen2,qwen3,qwen3_5,llama,mistral,gemma2,gemma3,phi3,granite,smollm3,lfm2}.py`, `model_import/common.py`, tokenizer/generation configuration, normalization/linear/embedding/RoPE/DeltaNet/short-convolution contracts, `weight_matrix.py`, `compressed/{mlx,quant_params}.py`, `compressed/utils/packing.py`, and `safetensors.py`. The catalog records the registry-code revision and hashes of inline metadata fragments. Source receipts record checkpoint-file revisions and content hashes. Converter receipts recursively hash conversion-affecting package resources and the exporter, with the dependency lock recorded separately.

The replacement target is independently qualified support for every in-scope pinned inventory row. There is no permanent fallback. Unsupported rows remain visible until their implementation and acceptance gates pass. Public package publication and release infrastructure require a separate scoped task.
