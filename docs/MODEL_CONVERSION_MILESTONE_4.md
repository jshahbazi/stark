# Milestone 4: dense hybrid architectures and future checkpoints

Date: 2026-10-12. Branch: `codex/conversion-milestone-4`. Incremental baseline: `2b47515b4a6b78ae7c81f52e9b4f5cff6d3ac685`. Package version: **0.6.0**. Reference: clean Lalamo [`831accb8252a6b6606c378048caeb44178f1a050`](https://github.com/trymirai/lalamo/tree/831accb8252a6b6606c378048caeb44178f1a050). The [plan](MODEL_CONVERSION_PLAN.md), [format inventory](MODEL_CONVERSION_FORMAT_INVENTORY.md) and [independent review](MODEL_CONVERSION_MILESTONE_4_REVIEW.md) define the scope and remaining gates. Earlier qualification and benchmark documents retain their original sources/results.

## Delivered scope and coverage

Independent dense adapters now convert Qwen3.5 text/DeltaNet and LFM2 short-convolution/attention. Selected `Qwen/Qwen3.5-0.8B` and `LiquidAI/LFM2-350M` BF16 outputs pass complete external tensor/spec/config/tokenizer parity and named-build native readiness followed by stdin EOF, with **zero scoring or generation calls**. The writer and Rust/Metal engine are unchanged. The runtime converter neither imports nor invokes Lalamo, JAX or PyTorch, and has no fallback.

| Coverage measure | Baseline | Final |
| --- | ---: | ---: |
| Pinned built-ins | 110 | 110 |
| Implemented built-ins | 54 | 63 |
| Implemented dense built-ins | 33 | 42 |
| Implemented MLX built-ins | 21 | 21 |
| Separate implemented Stark extensions | 4 | 4 |
| Exact BF16 named-build load-only tuples | 19 | 21 |
| Planned built-ins | 56 | 47 |
| Implemented identities without exact native qualification | 39 | 46 |

The nine added built-ins are dense `Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B}` and `LiquidAI/LFM2-{350M,700M,1.2B,2.6B}`. Only the selected 0.8B/350M tuples gain native qualification. Other sizes/config variants remain implemented but unqualified. Qwen3.5/LFM2 MLX, Qwen3.5 MoE, later generations, full multimodal conversion/execution and Mamba/LLaMBA remain outside this packet.

The catalog exporter explicitly excludes Qwen3.5-35B-A3B from dense promotion. The shared pinned config class handles both dense and MoE; its class name alone was insufficient classification. The [official checkpoint config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json) declares `qwen3_5_moe` and 256 experts. Config SHA-256 is `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`. Independent inspection corrected this overclaim before any timed run. A regression fixes the exact promoted identities and keeps MoE/later generations planned.

## Architecture reuse and checkpoint tracking

| Record | Purpose | Qualification boundary |
| --- | --- | --- |
| Architecture adapter | Config semantics, tensor names/shapes, transforms, input consumption and weight format | A supported architecture still requires complete config/layout validation |
| Checkpoint catalog profile | Metadata routing, reviewed parser/reasoning/defaults and declared packing | Applies only to its registered identity and accepted variants |
| Private source manifest | Immutable repository revisions, selected files/content hashes and replay selection | Binds one complete primary/external source set |
| Native qualification ledger | Exact repository/dtype/source/output/config/tokenizer hashes plus native build | Load readiness for the recorded tuple/build only |

Future fine-tunes can reuse any of the eleven supported architecture adapters through explicit `--metadata-profile hf-generic`. The checkpoint's canonical `model_type` selects the adapter; optional `--architecture` asserts it. No catalog entry is required for a new identity. This retains strict semantic fields, source/tensor consumption and supported format validation. It does not expand the 110-entry denominator or infer compatibility from a model name.

Generic metadata is self-contained: model config, tokenizer and tokenizer config are required; generation metadata and Jinja template files are selected whenever present. A supported template and nonempty explicit EOS identifiers are required. Identical duplicate inline/Jinja templates are accepted; conflicting templates fail. Plain roles are used without another checkpoint's reasoning/parser/system defaults. Existing catalog profiles reject generic overrides. Generic schema-2 replay binds architecture/profile, metadata roles and optional-sidecar presence; catalog schema-1 identities remain unchanged.

Generic receipts expose architecture, profile and `catalogued=false`. BF16 artifacts remain native-unverified unless a runtime limitation blocks them; F16/F32 remain conversion-only. `--require-native-support` refuses unregistered tuples. The generic path includes existing affine MLX 4/8-bit support for Qwen3, Llama, SmolLM3 and Gemma3 text, with the same strict plane/bit/group checks. New architectures and unsupported semantic/layout variants still require code and evidence. See [commands and requirements](../tools/converter/README.md#future-checkpoints-and-architecture-tracking).

## Adapter behavior and source integrity

Qwen3.5 merges nested text configuration using the pinned top-level override precedence, including EOS metadata. It preserves split/direct/group-interleaved DeltaNet projection ordering, gated full attention, rotary/norm/MLP settings and F32 recurrence/convolution parameters. A_log remains the stored F32 parameter; the native DeltaNet contract applies its transform. Contiguous bounded row views preserve original source signatures and reuse shared writer operations.

The real wrapper contains **488 input tensors** and **1,746,882,752 source tensor bytes**. Text conversion omits **153 declared vision tensors and 15 declared MTP tensors**, retaining **230 output tensors** and **1,505,783,040 payload bytes**. Omission requires the declared wrapper/positive MTP layer count and exact accepted prefixes. Undeclared or unexpected tensors fail. All original input files remain in initial/final complete content and signature checks, including omitted tensors. The selected decoder has eighteen DeltaNet and six full-attention layers; its 128-dimensional DeltaNet heads and kernel length four satisfy the current native load contract.

LFM2 preserves its attention/short-convolution layer order, head/Q/K normalization, accepted 2D/3D convolution shapes, feed-forward dimensions and gated projection order. F32 normalization/convolution tensors and BF16 projections retain pinned exports. The real checkpoint has **148 input tensors**, **708,967,936 source tensor bytes**, and **120 output tensors**. All inputs are consumed, with tied-head duplicates checked before omission where applicable.

Both adapters support dense BF16/F16/F32 conversion through bounded copy/cast/fusion operations. Unsupported quantization, MoE, malformed layer/head/group dimensions, biases or semantic options fail before publication. DeltaNet native dimensions other than 128 are reported runtime-blocked; conversion does not imply another native kernel contract. Source rechecks, sanitized failures, private staging and atomic no-replace publication remain enabled.

## Exact selected qualification

| Selected BF16 tuple | Primary revision | Source-manifest SHA-256 | Tensors |
| --- | --- | --- | ---: |
| `LiquidAI/LFM2-350M` | `f37d3f5c8c5484bc01dad379a595cf4c68c4e70e` | `3e3223d644f3ed803e460198d2c7890825669f509ef99b711881e233ef6ddafe` | 120 |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | `12027dc95b6c15c7ee094840632a9f290315c60cabec888afeda823c460356c3` | 230 |

LFM2 uses its own tokenizer/Jinja template and the exact reviewed inline native-generation profile. Qwen3.5 uses its own tokenizer/template and independently pinned generation metadata from `Qwen/Qwen3.5-27B` revision `fc05daec18b0a78c049392ed2e771dde82bdf654`, content hash `303aba891d66ab63908a7b3cc9163bcb835fdf8b9f6301c73216f3f1eb3992dd`. Its reviewed inline overrides remain role/hash bound. All metadata matches the actual registered external reference, including model-specific reasoning/parser defaults.

| Artifact | LFM2-350M SHA-256 | Qwen3.5-0.8B SHA-256 |
| --- | --- | --- |
| Stark weights | `d4e548ab46e52fc5cae93c11e293b02dc71a7166edc8084c18524ba66b97be1c` | `edcf1e34141a4fff5682d9bac2ed0c76af1970ab8f45bd2224126ae06f113403` |
| Root config | `79c5a52f032c816ef725f3e15f709670e4c91c2106251c30e56d2aabd2b0b682` | `6f0d92a7057105d6d524378f25b7c60214a3bc29013187b42c16eaf6d434ec16` |
| Tokenizer | `98cff83b4f6d7e9d8929bebc62b07e92cf1b3f99c80d16bafe8b84a75448f40b` | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| Frozen Lalamo weights | `d4aa021ab9aa5c210734f21162f55a4b135debf2db80290fe2970f97b55a5d80` | `ff75599bd518b3a1f0dcdffba9bc1e895a937bf8c5069731137259a9c219581e` |
| Private load-only qualification evidence | `ac9a3c71c86a683239f1cbec188e15ef446af4c9ad3cd86869625ac7944a6aef` | `ecea6f37f98232b900dccaf90249d0a0139c5c24c246fbca005f716e1273aad2` |

Different safetensors serialization order explains differing reference/candidate whole-file hashes; every tensor name, shape, dtype, payload hash, spec/logical field, complete config and normalized tokenizer matches. Candidate artifacts must retain their exact qualified whole-file hashes. Prior nineteen native records are unchanged.

The native build is Rust **1.98.1**, release, `default` plus `optimized` features. Worker SHA-256 is `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d`; native source manifest is `0cc926bcf73db3921eab03f127185626b7f4311b73c62dad79b407c642d0c9af`; Cargo lock is `65444f3e0e0707c43521539e43c95365cfc6a51ec6f9e90180bc43bc8fb4603e`; Rust toolchain file is `72da9f9e503464ba7393560f6f51a69fb06be72633d52ae7725fa0ba59dfda22`. Readiness and EOF establish loading for these exact artifacts/build. Scoring, generation, inference speed and quality remain unverified for these tuples.

## Complete-command benchmark

| BF16 output | Stark median [min–max] | Lalamo median [min–max] | Median RSS Stark / Lalamo | Raw latency hypothesis |
| --- | ---: | ---: | ---: | --- |
| LFM2-350M | 1.233 s [1.127–7.704] | 1.688 s [1.417–7.082] | 114.83 / 1168.86 MiB | Passed; 26.95% lower median time |
| Qwen3.5-0.8B | 3.525 s [2.487–5.327] | 2.707 s [2.608–6.831] | 201.52 / 2348.45 MiB | Failed; 30.21% higher median time |

All twenty outputs pass exact external parity; all ten candidate receipts match the exact stored native qualification and final source/lock hashes. Both tuples use 8 MiB peak transformation workspace. Maximum sampled candidate RSS is 215,924,736 bytes. Median sampled RSS is **90.18% lower for LFM2** and **91.42% lower for Qwen3.5**, passing both memory hypotheses. LFM2's complete-command ratio is 1.37×. Qwen3.5 takes 0.818 s more at the median; its raw latency objective remains open.

Measurements ran on one arm64 host, macOS 26.7.1, 18 logical CPUs, 128 GiB RAM and Python 3.13.13, from 2026-10-12 05:45:00 to 05:47:10 UTC. Every preregistered observation is retained. Ranges are broad and overlap for both tuples; these descriptive medians do not establish statistical significance, another cache regime/host or general family-wide speed. Earlier LFM2 observations were much slower for both backends. No observation was dropped and no favorable rerun substituted.

The preregistered comparison runs **twenty serial fresh complete processes**: two BF16 tuples, candidate/live pinned Lalamo, five repetitions each. It alternates backend order and reverses the starting order for the second tuple. Selected files are warmed before each run. Setup/download, warming, native loading and parity comparisons are excluded from timing. Candidate commands include source verification, metadata normalization, output writing, inline hashes, final complete verification, receipts and `--require-native-support`. Lalamo performs weaker source/receipt verification. No prior implementation baseline exists for these newly supported architectures.

Hard acceptance requires unchanged frozen sources/tooling/reference outputs, full parity, exact native qualification for every candidate, transformation workspace at most 8 MiB and sampled candidate process-tree RSS below 1 GiB. Raw median latency no slower than Lalamo and median RSS at most half of Lalamo are separate hypotheses; a failed timing hypothesis remains visible. RSS is sampled every 10 ms, can miss peaks/count shared pages more than once, and is not an enforced total-process cap. Timed redundant model copies are deleted only after private receipts/hashes/parity have been preserved; frozen references and qualified artifacts remain private.

## Validation and source bindings

The integrated final suite passed **1,977 tests**, with **46 skipped**, using twenty-one supplied private metadata reference profiles. The new external-export fixtures each contain sixty-three actual pinned full-decoder/export cases across nine source/target precision combinations and config/layout variants. Focused Qwen3.5 and LFM2 suites passed 102 and 186 tests before integration. Generic integration tests exercise all eleven architecture routes, existing MLX 4/8-bit routes, private offline replay, metadata requirements, optional-sidecar changes and identity/profile/role/path spoof rejection. Independent implementation review passed 566 targeted tests before the final catalog correction; its final disposition is recorded separately.

Ruff/check/format, locked dependency consistency, repository policy, seventeen CI-policy tests and diff checks passed. The cached optimized Rust 1.98.1 build passed with its existing debug-info stripping warning. No dependency version changed; the managed converter runtime remains Python 3.13.13 with 27 locked packages. No hosted, advisory-scan or inference-evaluation result is claimed.

| Final source/evidence artifact | SHA-256 |
| --- | --- |
| Converter sources | `27cc315ef9dff71fba085dfefbfd136bc754512a72a9ceb0a1e6b8158f9d18bd` |
| Converter lock | `3887b0b9c6814edd02bf2907c99921be4aba91eb3739ea6bc8f97a39181c38c4` |
| Corrected 110-entry catalog plus four extensions | `c9fb140393e39c7818759db3434f67c524f868583019e8dc3a3463b60cf2dbbe` |
| Twenty-one native qualification records | `fefb5bcd889be5f09b94b675bb717097d4ba89a5d5dec1853f6c5bb7c25b1eec` |
| Preregistered objectives | `d467a026b5c2bd00271bebbf06a37852f16930dd667ff9d46d03b987dc7b48bc` |
| Final benchmark freeze | `d96499477fa1a400301c03684ded73105e0fa30a663bcd73f397cb74272b0516` |
| Twenty-run benchmark | `c0adf9f26b1d3bfcde0d4b6221c9ccae441956644658017299d2a698ebc7114d` |
| Benchmark coordinator | `eae624d56679e6de01125829e139cc8d8040470a6e475230895e5c16f3b827c8` |
| External reference runner | `e785c2272366f523f79237be847549f76ea07bdc44b20cf69b7d76b6e38ad010` |
| External reference preparation | `6d801ee24e66803a8c0b669d40a063cc48d708d863efb6d3c18b2baa2fed8f5f` |
| Native load-only coordinator | `1de58465fc5229d89dd2f7a208d3f5e58b4c610333f795bd85593e32ab96b2dc` |
| Qwen3.5 frozen decoder/export cases | `e71064e5116d00e163a9d4466aab8ba6ab055ad20bfad693de16e2502e8a6dc5` |
| LFM2 frozen decoder/export cases | `434eb1a066c61f01420674b7e39d72e245aacb35e8c78ab2970607bd0f897315` |

The original load-only evidence binds pre-ledger converter source `ae06291db5adc2bbd5c19356090d0a635e1d925f30f06e268b75aa09a79ee84f` and catalog `5ae613a7a37b7441e9da1e955fdc02dcdebaf189b6a0fb96bf0bbd432e4ecb44`. The final benchmark separately binds the twenty-one-record ledger and corrected catalog/source. No earlier evidence was reassigned to newer code. A pre-correction freeze (`30640f76e6a5337dd67bbb1987ba79cd1c02a95f940ead28885a1fb79af82e7b`) is retained privately and produced **zero timed runs**.

## Remaining work and provenance

Next is the independent Qwen3.5 text/LFM2 MLX continuation, with separate bit/group/config/metadata/native qualification and benchmarks. The [format inventory](MODEL_CONVERSION_FORMAT_INVENTORY.md) distinguishes actual AWQ/MLX/MXFP4 loaders from output types and enum-only descriptions. Complete G4/G5, remaining specialized/classifier/TTS/MoE/NeMo rows and unmeasured variants remain open. SmolLM3 8-bit latency is deferred by user choice and is not a dependency of this packet.

Adapters/tests were built in parallel with disjoint ownership. A separate reviewer authored no production implementation or qualification/benchmark coordinator. Final review recomputes aggregate evidence and checks exact preservation of old records. All real weights, configs, tokenizers, operational manifests, reference outputs and native artifacts remain ignored/private; public artifacts contain aggregate outcomes, public source identities and hashes. No Jev/JevK5 run, scoring/generation experiment, distribution or inherited cloud/signing/release change occurred.

Adapted pinned sources include `lalamo/model_import/model_configs/huggingface/{qwen3_5,lfm2}.py`, `model_import/loaders/huggingface.py`, the Qwen/LFM model specs, generation/tokenizer metadata contracts and layer export semantics. [Lalamo's MIT license](../tools/converter/LICENSE.lalamo) and Mirai copyright are preserved. Stark retains Uzu base revision `8d01581474760bd1fc7f4a547ff1d02f7dada437`.
