# Stark model converter

`stark-convert` is the repository's independent, bounded-memory converter for selected Hugging Face checkpoints. It produces Stark's existing `config.json`, `tokenizer.json`, and `model.safetensors` files. The runtime converter uses registered streaming adapters and has no Lalamo, JAX, or PyTorch dependency. Unsupported combinations fail explicitly; there is no upstream fallback.

The [replacement plan](../../docs/MODEL_CONVERSION_PLAN.md) targets the complete built-in inventory at the pinned Lalamo reference revision. [Milestone 3 dense MLX evidence](../../docs/MODEL_CONVERSION_MILESTONE_3_DENSE.md) records Llama, SmolLM3 and Gemma text conversion and performance. The [first MLX packet](../../docs/MODEL_CONVERSION_MILESTONE_3.md), [milestone 2](../../docs/MODEL_CONVERSION_MILESTONE_2.md) and [milestone 1](../../docs/MODEL_CONVERSION_MILESTONE_1.md) retain their original source versions and results.

## Convert a model

Run from the repository root with managed Python 3.13.13 through `uv`:

```sh
uv sync --project tools/converter --locked
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --write-source-manifest runtime/converter-sources/qwen3-06b.json \
  --output models/qwen3-06b-stark
```

Remote inputs resolve primary and required external repositories to immutable Hugging Face commits before downloading. Authentication uses Hugging Face's existing credential mechanism, and downloads stay in its local cache. Local checkpoints require their registered profile identity:

```sh
uv run --project tools/converter --locked stark-convert /absolute/path/to/checkpoint \
  --repo Qwen/Qwen3-0.6B --output models/local-qwen3-stark
```

`--backend auto` and `--backend streaming` both use the independent implementation. BF16 is the default output dtype. All implemented dense adapters perform bounded BF16/F16/F32 conversion with frozen reference-byte tests. Mixed source dtypes within a fused group fail explicitly. Existing output directories are refused. The converter stages with owner-only permissions, validates complete supported layouts, and publishes using an atomic no-replace rename on macOS or Linux. Failed conversions remove staging output.

The command emits aggregate timings, hashes, and capability status. Its private `conversion.json` receipt binds source revisions/content hashes, converter inputs, the dependency lock, emitted files, and transformation workspace use. Errors expose fixed categories without exception text or native diagnostics.

## Catalog coverage and qualification

The attributed [catalog](stark_convert/catalog.json) accounts for **110 built-in reference entries**, with **four explicitly separated Stark extensions**. Implemented adapters and metadata profiles cover **54 built-ins plus four extensions**, or 58 identities. This includes 33 dense built-ins and 21 MLX built-ins:

| Adapter | Implemented identities |
| --- | --- |
| Qwen3 | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-4B-Thinking-2507`, `POLARIS-Project/Polaris-4B-Preview` |
| Qwen3 MLX | `Qwen/Qwen3-{0.6B,1.7B,4B,8B,14B,32B}-MLX-{4bit,8bit}`; affine groups 16/32/64/128, divisible logical input widths |
| Qwen2 | `Qwen/Qwen2.5-Coder-{0.5B,1.5B,3B,7B,14B,32B}-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
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

This is implementation coverage. Exact checkpoint/dtype/config qualification remains per tuple and is recorded in the milestone evidence. A passing small checkpoint does not qualify every size or config variant in its family. Other catalog entries, compressed schemes, custom architectures/RoPE variants, and unsupported metadata representations fail with a sanitized unsupported category. Arbitrary Hugging Face repositories are outside the current catalog.

MLX conversion repacks existing affine checkpoints without computing new quantization. Little-endian packed I32/U32 words become U8 tensors with identical payload bytes. Projection scales and affine biases are cast, fused, transposed, and padded to Stark's group/output layout; input and tied embedding planes retain their input/output layout. A complete plane triplet is required for every compressed matrix. Fused groups must share representation and coefficient dtype. The catalog binds each compressed identity to its declared 4-bit or 8-bit format, and contradictory declarations or layouts fail.

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen3-0.6B-MLX-4bit \
  --revision 173234aa840d113125e9f2271100ddbaf16c9620 \
  --write-source-manifest runtime/converter-sources/qwen3-06b-mlx4.json \
  --output models/qwen3-06b-mlx4-stark
```

All MLX families use affine groups 16/32/64/128 and the same bounded coefficient operations. Llama preserves its accepted Llama 3 RoPE; SmolLM3 preserves its per-layer RoPE mask, omitted tied-head default and accepted projection biases. Gemma text retains runtime embedding scaling, normalization offsets and local/global attention/RoPE semantics. Quantized Qwen2, Mistral, Granite and Gemma2 retain their unsupported status.

MLX metadata follows each registered source rule: Qwen3 uses its original repository's tokenizer/generation files; Llama uses the MLX tokenizer and official generation file; SmolLM3 uses its MLX tokenizer and original chat-template/generation files; Gemma text uses its MLX tokenizer and official generation file. External sources are independently pinned. Missing or gated dependencies fail rather than silently replacing metadata. An explicit private offline manifest can reference verified local content; the separate qualification evidence records any public mirror transport and its exact canonical Git-blob match.

The Qwen3-0.6B group-128 BF16 tuples and selected Llama-3.2-1B, SmolLM3-3B and Gemma3-1B group-64 BF16 tuples have exact artifact parity and named-build load-only qualifications at both bit widths. Other sizes and groups remain unqualified. No compressed inference-speed or quality claim is made.

Probe the catalog without reading or downloading a checkpoint:

```sh
uv run --project tools/converter --locked stark-convert Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --capabilities
```

The probe reports catalog implementation, model kind, and runtime status with `source_validation=not_performed`. Successful conversion validates the complete accepted source layout. Current implemented adapters emit language-model root configs. Classifier, TTS, specialized state-space/hybrid, MoE, and other compressed conversion remain later workstreams.

Stark's supported dense loader selects BF16 weights. F16/F32 outputs are conversion-only and are marked runtime-blocked. The catalog probe remains unverified because it does not inspect source or output artifacts. A BF16 conversion reports `supported_layout_unverified_tuple` unless its registered identity, source-manifest hash, weights hash, config hash, and tokenizer hash exactly match a stored native qualification.

`--require-native-support` requires that exact stored match before publication; unmatched artifacts fail cleanly. A match reports `native_loading=qualified_tuple` and emits the full `native_qualification` record, including the named build's worker/source/Cargo-lock/toolchain hashes, Rust 1.98.1, release profile, features, and load-only result. That result applies to the recorded build and exact artifacts. It provides no guarantee for another current or future engine binary. Execution remains unverified: readiness and EOF do not establish scoring, generation, classification, or TTS correctness. Stark currently rejects MoE construction.

## Measured improvements over Lalamo

The current Llama/SmolLM3/Gemma MLX packet passed exact parity and BF16 native load-only checks for six group-64 tuples. It reduced median sampled process-tree RSS by **90.7–96.6%**. The bounded writer copies packed codes directly and exports coefficient tiles without allocating complete unpacked weight matrices. The installed runtime has no Lalamo, JAX or PyTorch dependency. Source/content verification remains complete.

| MLX checkpoint | Stark complete median | Lalamo complete median | Sampled RSS reduction |
| --- | ---: | ---: | ---: |
| Llama3.2-1B 4-bit | 1.845 s | 2.292 s | 94.1% |
| Llama3.2-1B 8-bit | 2.233 s | 2.137 s | 96.6% |
| SmolLM3-3B 4-bit | 3.979 s | 2.809 s | 95.4% |
| SmolLM3-3B 8-bit | 6.019 s | 3.544 s | 96.2% |
| Gemma3-1B 4-bit | 1.991 s | 2.444 s | 90.7% |
| Gemma3-1B 8-bit | 2.655 s | 2.171 s | 92.5% |

Llama 4-bit and Gemma 4-bit measured 1.24× and 1.23× complete-command speedups. The other four median timings are slower, and Llama 8-bit timing ranges overlap. Broad latency parity is still open. The earlier Qwen3-0.6B 4/8-bit packet measured 1.78×/1.21× at its own source state. Ratios are conversion measurements only and grant no inference or quality qualification.

Each backend ran three serial fresh processes per tuple with warm selected files and alternating order. Setup/download, warming and parity comparisons were excluded. Complete commands include verification, metadata, writing, hashes and receipts; Lalamo performs weaker source/receipt verification. RSS was sampled every 10 ms and is not a total-memory ceiling. Exact sources, timing ranges, host versions, failed hypotheses and independent review are in the [current evidence](../../docs/MODEL_CONVERSION_MILESTONE_3_DENSE.md).

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

Reported complete CLI time includes startup, conversion, verification, and receipt hashing. Process-tree RSS is sampled every 10 ms; shared pages can be counted more than once and brief peaks may be missed. The harness rejects source, reference, and converter-input changes during the run. Measurements run serially with other model-dependent work paused.

The optional metadata parity tests consume supplied private artifacts through `STARK_CONVERT_REFERENCE_ARTIFACTS`; they never install or invoke Lalamo. Each artifact profile uses `<root>/<repo with slash replaced by __>/provenance.json`, `sources/<repo>/...`, and `expected/config.json` plus `expected/tokenizer.json`. The provenance field `reference_revision` must equal the pin below. Unsupplied profiles remain unqualified.

Models and permitted native aggregate/hash validation artifacts remain in ignored private storage. Only load-only readiness/EOF aggregates and hashes were retained. Publish aggregate outcomes and hashes only. Do not capture prompts, responses, token vectors, decision probabilities, credentials, or exception dumps. The [historical prototype evidence](../../docs/MODEL_CONVERSION.md) retains its original implementation and lock hashes; its timing ratios apply to that prototype only.

## Provenance and replacement policy

Lalamo serves as a pinned design and qualification reference at [`831accb8252a6b6606c378048caeb44178f1a050`](https://github.com/trymirai/lalamo/tree/831accb8252a6b6606c378048caeb44178f1a050). The catalog exporter reads that source in a separate developer environment. Runtime conversion reads the checked-in declarative catalog and uses Stark implementations. [Lalamo's MIT license](LICENSE.lalamo) and Mirai Tech Inc. copyright are preserved. Stark retains Uzu base revision `8d01581474760bd1fc7f4a547ff1d02f7dada437`.

Adapted source paths include `model_import/model_spec.py`, `model_import/model_specs/`, `model_import/origins.py`, `model_import/loaders/huggingface.py`, `model_import/model_configs/huggingface/{qwen2,qwen3,llama,mistral,gemma2,gemma3,phi3,granite,smollm3}.py`, `model_import/common.py`, tokenizer/generation configuration, normalization/linear/embedding/RoPE contracts, `weight_matrix.py`, `compressed/{mlx,quant_params}.py`, `compressed/utils/packing.py`, and `safetensors.py`. The catalog records the registry-code revision and hashes of inline metadata fragments. Source receipts record checkpoint-file revisions and content hashes. Converter receipts recursively hash conversion-affecting package resources and the exporter, with the dependency lock recorded separately.

The replacement target is independently qualified support for every in-scope pinned inventory row. There is no permanent fallback. Unsupported rows remain visible until their implementation and acceptance gates pass. Public package publication and release infrastructure require a separate scoped task.
