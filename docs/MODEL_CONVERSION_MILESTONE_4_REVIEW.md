# Independent milestone 4 technical review

Review completed: 2026-10-12. Incremental baseline: `2b47515b4a6b78ae7c81f52e9b4f5cff6d3ac685`. Branch: `codex/conversion-milestone-4`. This review covers the [dense hybrid architecture and future-checkpoint packet](MODEL_CONVERSION_MILESTONE_4.md). The reviewer authored no production converter, test fixture generator, qualification helper or benchmark coordinator. Earlier source-bound results retain their original evidence.

## Disposition

**Accept the scoped dense Qwen3.5/LFM2 adapters, explicit future-checkpoint conversion, selected exact artifact parity, named-build load readiness and bounded transformation workspace.** No unresolved blocking finding remains. All twenty measured outputs pass external tensor/spec/config/tokenizer parity. All ten candidate conversions match the two newly recorded exact BF16 qualifications. The prior nineteen qualification records are preserved unchanged.

Qwen3.5-0.8B fails the preregistered raw latency hypothesis: its complete-command median is **30.21% higher than Lalamo's**. LFM2-350M passes with a **26.95% lower median**. Both memory hypotheses pass. The broad observed timing ranges overlap for both tuples; these descriptive results establish no statistical significance or general latency guarantee. Qwen3.5 latency remains open work. Full G4/G5, new-family MLX conversion and complete Lalamo replacement remain open.

## Reviewed bindings

| Artifact | SHA-256 |
| --- | --- |
| Final converter sources | `27cc315ef9dff71fba085dfefbfd136bc754512a72a9ceb0a1e6b8158f9d18bd` |
| Final converter lock | `3887b0b9c6814edd02bf2907c99921be4aba91eb3739ea6bc8f97a39181c38c4` |
| Corrected catalog | `c9fb140393e39c7818759db3434f67c524f868583019e8dc3a3463b60cf2dbbe` |
| Twenty-one-record native ledger | `fefb5bcd889be5f09b94b675bb717097d4ba89a5d5dec1853f6c5bb7c25b1eec` |
| Qwen3.5 actual decoder/export fixture | `e71064e5116d00e163a9d4466aab8ba6ab055ad20bfad693de16e2502e8a6dc5` |
| LFM2 actual decoder/export fixture | `434eb1a066c61f01420674b7e39d72e245aacb35e8c78ab2970607bd0f897315` |
| Private Qwen3.5 fixture generator | `0bf36e7ddb4e6d4021f76c14c293c0b5a2aae07116b8356e6fea2fd78e408b42` |
| Private LFM2 fixture generator | `23067876d3de3c2e08028beb34ddac492a370bffaf10d86bbc80ed9a8bdbaa15` |
| LFM2 load-only qualification evidence | `ac9a3c71c86a683239f1cbec188e15ef446af4c9ad3cd86869625ac7944a6aef` |
| Qwen3.5 load-only qualification evidence | `ecea6f37f98232b900dccaf90249d0a0139c5c24c246fbca005f716e1273aad2` |
| Native qualification helper | `1de58465fc5229d89dd2f7a208d3f5e58b4c610333f795bd85593e32ab96b2dc` |
| Reference preparation evidence | `6d801ee24e66803a8c0b669d40a063cc48d708d863efb6d3c18b2baa2fed8f5f` |
| Preregistered benchmark objectives | `d467a026b5c2bd00271bebbf06a37852f16930dd667ff9d46d03b987dc7b48bc` |
| Final benchmark freeze | `d96499477fa1a400301c03684ded73105e0fa30a663bcd73f397cb74272b0516` |
| Twenty-run complete-command benchmark | `c0adf9f26b1d3bfcde0d4b6221c9ccae441956644658017299d2a698ebc7114d` |
| Benchmark coordinator | `eae624d56679e6de01125829e139cc8d8040470a6e475230895e5c16f3b827c8` |
| External reference runner | `e785c2272366f523f79237be847549f76ea07bdc44b20cf69b7d76b6e38ad010` |
| Named optimized worker | `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d` |
| Native source manifest | `0cc926bcf73db3921eab03f127185626b7f4311b73c62dad79b407c642d0c9af` |

The reviewer independently recomputed these hashes, the selected private source-manifest bindings and retained candidate/reference artifact hashes. Final source/lock/native-build identities match the freeze and benchmark's post-run identities. Fresh comparisons of both retained complete artifact sets reproduce their original qualification parity records. Every timed candidate's source, weights, config and tokenizer hashes match the exact qualification matcher and its named native build.

The ledger comparison against the baseline preserves all nineteen earlier records, including their nested build evidence. All catalog identities, source FileSpecs and checkpoint metadata definitions are preserved; only implementation/status routing is extended. Parsing both locks confirmed unchanged dependency records and a sole editable package version change from 0.5.1 to 0.6.0.

The original two load checks retain their pre-ledger converter source binding. The final benchmark separately binds the corrected catalog and final twenty-one-record ledger. A pre-correction freeze remains private and produced zero timed observations. These distinctions preserve evidence provenance without extending an earlier run to later code.

## Architecture, metadata and replay

Conversion is reusable by architecture and input format. Checkpoint profiles retain reviewed metadata/source rules; qualifications retain exact repository, dtype, complete source/artifact hashes and build identities. The catalog exporter promotes five dense Qwen3.5 and four dense LFM2 identities. The shared Qwen3.5 config class also handles MoE, so promotion now uses the explicit dense identities. Qwen3.5-35B-A3B, later Qwen/LFM generations and new-family MLX profiles remain planned. Independent final registry tests verify this boundary. Coverage is 63 implemented built-ins plus four extensions; the pinned denominator remains 110.

Future identities require explicit `--metadata-profile hf-generic`. The checkpoint's canonical `model_type` chooses a supported adapter; an optional architecture assertion must agree. The planner still validates config semantics, dimensions, tensor shapes/dtypes and input consumption. Generic mode cannot override a registered profile's source routing, packing, reasoning or defaults. Source-free capability probes require an explicit architecture and report that source validation has not occurred.

Generic metadata reads the source's tokenizer, supported template and explicit generation fields. It uses plain roles and no inferred reasoning controls, output parser or default system prompt. Missing or empty EOS identifiers fail. Optional generation/Jinja sidecar presence is pinned; duplicate template forms must match exactly. Schema-2 replay binds the architecture/profile and validates metadata roles, paths, root bindings, external-source exclusion and optional selection. Catalog schema-1 receipt identities remain unchanged.

Independent synthetic checks exercised top-level Qwen wrapper EOS precedence through the complete conversion, including stop IDs and decoded codec EOS. A consistently forged known-repository schema-2 manifest, with primary/source/file identities all changed together, was rejected. Existing negative tests cover architecture/profile disagreement, optional-sidecar changes, role/path/source spoofing, malformed metadata, complete content mutations, sanitized failures, staging cleanup and atomic no-replace publication.

Generic BF16 output remains native-unverified; unsupported native dimensions and F16/F32 requests remain runtime-blocked. Generic `--require-native-support` fails. A registered exact artifact match reports the recorded named-build qualification only. Architecture support or successful conversion grants no future-checkpoint execution qualification.

## Adapter and native contracts

Qwen3.5 follows the pinned top-level-over-nested-text config precedence. Its contiguous row views preserve direct/split/group-interleaved DeltaNet projections and gated full-attention ordering while reusing bounded shared copy/cast/fusion operations. F32 convolution, normalization, A_log and dt_bias exports match the reference. The frozen actual-loader fixtures exercise all nine BF16/F16/F32 source/target pairs, gated projection/bias layouts, wrapper extraction and declared MTP omission.

Text conversion omits the exact `model.visual.` prefix only for a declared wrapper and the exact `mtp.` prefix only with a validated positive integer MTP layer declaration. Undeclared MTP, invalid declarations, alternative prefixes and unrelated extras fail. Every original tensor header/signature and complete source file remains verified, including omitted tensors and separately sharded auxiliary weights. Mutation tests confirm that omitted auxiliary shards still invalidate output. This grants text conversion only; vision/MTP/speculative execution remains outside scope.

LFM2 validates explicit short-convolution/attention topology, consistent layer indices, head/group dimensions, feed-forward rounding, RoPE declarations and accepted convolution aliases/shapes. Unsupported SwiGLU/position options, dimension disagreements, ambiguous aliases, unknown tensors, mixed fusion dtypes and packing fail. Its frozen actual-loader fixtures verify sixty-three full decoder exports, including tied/untied embeddings, convolution bias, 2D/3D source shapes, layer masks and automatic feed-forward dimensions.

The native loader currently requires DeltaNet key/value head dimensions of 128 and convolution kernel lengths of at least two. The CLI reports incompatible DeltaNet dimensions as runtime-blocked; the selected real checkpoint satisfies these dimensions. Both adapters retain F32 inner parameters and BF16 outer projection semantics required by the named loader. Neither the shared writer nor Rust/Metal engine changes in this packet. Full source verification, exact qualification checks, private staging and atomic publication remain enabled. Pinned upstream attribution and MIT license remain preserved; the runtime gains no Lalamo dependency or fallback.

## Validation and measured results

The reviewer independently ran **566 integrated focused tests, with 49 skipped**, covering generic conversion, sources, CLI, metadata, registry, qualification and both new adapters. They passed in 26.03 seconds before the final catalog correction. All **nine final registry tests** passed after that correction. Additional independent synthetic boundary checks and the final aggregate/hash audit passed. Both fixture generator hashes and Qwen's synthetic-input definition hash match their frozen fixtures. These generators actually invoke the pinned config, decoder loader and export functions; expected output is independent of the production planner.

The coordinator reports the final **1,977 passed and 46 skipped** with twenty-one supplied private metadata references, plus Ruff/format, lock consistency, repository policy, seventeen CI-policy tests and diff checks. The cached Rust 1.98.1 optimized release build passed with its existing debug-info stripping warning. The reviewer audited the two native readiness/EOF records and their unchanged binary/build inputs; the reviewer did not invoke a native model. Both actual records contain zero scoring calls. No inference-quality, hosted-CI or distribution result is claimed.

The benchmark runs five serial fresh direct Python processes per backend and tuple, alternating order and reversing the initial order across tuples. Every selected source file is warmed before each timed process. Startup, metadata, writing, output hashing, complete candidate verification, exact qualification lookup, receipt and final checks are timed. Setup/downloads, warming, native readiness and parity comparisons are excluded. External Lalamo is pinned at `831accb8252a6b6606c378048caeb44178f1a050` and performs weaker source verification and receipt work. These architectures had no prior Stark conversion baseline.

| BF16 tuple | Stark median [range] | Lalamo median [range] | Median sampled RSS Stark / Lalamo | Raw latency hypothesis |
| --- | ---: | ---: | ---: | --- |
| LFM2-350M | 1.233 s [1.127–7.704] | 1.688 s [1.417–7.082] | 114.83 / 1168.86 MiB | Passed |
| Qwen3.5-0.8B text | 3.525 s [2.487–5.327] | 2.707 s [2.608–6.831] | 201.52 / 2348.45 MiB | Failed |

The independent audit recomputed medians, complete ranges, ratios and hypothesis flags from all twenty observations. It checked all twenty parity records, all ten candidate workspace/RSS/source/tooling/artifact gates and exact qualification records, and the two retained actual native records. Median sampled RSS is 90.18% lower for LFM2 and 91.42% lower for Qwen3.5. Both pass the half-reference memory hypothesis, the 8 MiB transformation-workspace limit and sampled candidate tree RSS below 1 GiB.

RSS sampling every ten milliseconds can miss peaks and count shared pages more than once; this is an observed post-run check, not an enforced process cap. All timing outliers are retained, including the much slower early LFM2 observations on both backends. No observations were dropped or favorable reruns substituted. Five repeats on one warm-cache host grant no cold-cache/download result, statistical significance or family-wide speed claim.

## Remaining scope

Qwen3.5 complete-command latency, Qwen3.5/LFM2 MLX, other config/checkpoint/bit/group tuples and remaining specialized/classifier/TTS/MoE/container support retain separate implementation and qualification gates. SmolLM3 8-bit latency remains deferred by user choice. Load readiness for two exact BF16 artifacts does not establish inference correctness, throughput or compatibility with another engine build. Operational models, metadata, manifests, benchmarks and native artifacts remain private and ignored; this review publishes aggregate results, public identities and hashes only.
