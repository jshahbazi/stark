# Independent milestone 3 dense MLX technical review

Review completed: 2026-10-12. Implementation and benchmark evidence: 2026-10-11. Incremental baseline: `e40dc0ce` on `codex/conversion-milestone-3`. The reviewer authored no production implementation, source-recovery coordinator, or benchmark coordinator. This review covers the [Llama/SmolLM3/Gemma text continuation](MODEL_CONVERSION_MILESTONE_3_DENSE.md); the [initial Qwen3 review](MODEL_CONVERSION_MILESTONE_3_REVIEW.md) remains source-bound to its original packet.

## Disposition

**Accept the scoped independent conversion, six exact BF16 load-only qualifications, bounded workspace, and observed conversion-memory improvements.** No unresolved blocking finding remains. Nine additional pinned profiles are implemented: five Llama, two SmolLM3 and two Gemma3 text. Current coverage is 54 of 110 built-ins plus four extensions, including 21 MLX profiles. Nineteen exact artifact/source tuples have named-build loading evidence; the prior thirteen records are preserved unchanged.

The six-tuple complete-command latency hypothesis passes for Llama3.2-1B 4-bit and Gemma3-1B 4-bit. The other four measured medians are slower than the pinned external converter. This review accepts those recorded outcomes and the README's bounded claims. It does not grant general latency promotion, complete Phase 4/G4 acceptance, native execution/quality, or complete Lalamo replacement. Fifty-six built-ins remain unimplemented and 39 implemented identities lack exact native qualification.

## Reviewed source and evidence binding

| Artifact | SHA-256 |
| --- | --- |
| Final converter sources | `779ec68a7a1c55edff0438ebe60afdca7194e3e6046679491cac06bcc6656aa1` |
| Converter lock | `06d5e2a227ce9480adaa69501a6d7552c68c95e2d37dc0bd830d650ac98fae55` |
| Catalog | `d855e4c02ada48cca0062e7c273d3d7372b67b4ba70379ef3cf275acd9b78a3a` |
| Nineteen exact native qualification records | `26074c55ae93a5ef0a7aba15a676668ba4f0d4fa76466e23c2076b3515b8beac` |
| Final six-tuple benchmark | `576a22a4e1b5a90054ada0056bfab425f4b71d47de3657163c290a819770674d` |
| Qualification summary and final tooling | `9c863fa6f27cbacfce1ba47f8710897e2abdef7b8484e4f39c69c10ab06a6067` |
| Preserved initial four-tuple benchmark | `a226d597954df6fb2a39177ca074945844fa99c26a6c4b6df77db10aae70edd9` |
| Final benchmark coordinator | `1882623bfe69612feea28abbb7183e748bb840021cd201e83df0cd1c7bdbbae4` |
| External reference runner using manifest metadata paths | `e785c2272366f523f79237be847549f76ea07bdc44b20cf69b7d76b6e38ad010` |
| Llama/SmolLM3 frozen decoder exports | `f891c31a24442bc2db061bd60150c723613b22cc7bc79bfd1a7e5021b9d5697c` |
| Gemma frozen exports | `d350a885d8e52da04d28cc645e8bf493a9bd5dca42ae191c4bca4c13f08695a4` |
| Eight complete Llama/SmolLM3 synthetic comparisons | `c23b6e683c8b3b8aba746a88aaf5d1b66f09faa1b373a1b135a6e672a762feb7` |
| Thirteen complete Gemma synthetic comparisons | `a98d7c9821a6e429f54c7009ecd933e7129eec2b32986801e18d9a06c5b07e19` |
| Exact public mirror evidence | `3c9fb4f70658a7d8a2d9377b0b6348277825fc1d752494cd51f7a03ea27e047c` |
| Llama recovery preparation | `f4a2d0d3dde276fb22796c5ad4795cf3f4fa9df15add93fb991440d41d215bfd` |
| Private source-recovery script | `96713ed2c0818907370ad472e9d60b39242a337cc50c16dee767ce772b5979c2` |

The reviewer recomputed final source/lock, catalog, ledger, benchmark and fixture hashes. All 36 timed outputs have exact external parity; all 18 final-source candidate receipts match the exact stored qualification and current tooling hashes. The ledger preserves all thirteen baseline records unchanged. The new benchmark and qualification receipts remain separate from earlier results.

The initial synthetic and Smol/Gemma qualification packets retain source `d07bf3bbadab099c090a30630dab251c40065bdc5903382cde1696326689d9e5`. The four-record ledger update produced source `cfde8302d7532be773c19953f9ee45ab1cada07d8c20b5f7961f7f2aaa0485f9`, which binds the initial four-tuple benchmark and Llama qualifications. The final six-tuple benchmark binds the nineteen-record ledger and final source above. Prior measurements were not silently reassigned to newer code.

Reference semantics were inspected directly in the clean Lalamo checkout at `831accb8252a6b6606c378048caeb44178f1a050`, including its Hugging Face matrix loader, MLX export, and Llama/SmolLM3/Gemma3 text configs. The attributed MIT source conventions and `LICENSE.lalamo` remain intact. Package version 0.5.0 changes no dependency version. The converter runtime gains no Lalamo, JAX, PyTorch or Transformers import or invocation.

## Technical assessment

**Matrix representation and consumption.** The three new planner paths reuse the unchanged reviewed compressed helper. Accepted little-endian I32/U32 words become U8 payloads without unpacking a full matrix or computing new quantization. Each matrix requires complete weight, scale and affine-bias planes with matching logical dimensions, declared bits/groups, packing divisibility and coefficient dtype. Q/K/V and up/gate fusion preserve the pinned loader's row ordering across source shards. Projection/readout coefficient planes transpose and pad after the complete fused output count; input/tied embedding planes retain their input/output orientation. Optional linear bias vectors remain separate from affine bias planes.

Every accepted duplicate tied-head plane must match the corresponding embedding plane in dtype and complete payload hash. Incomplete duplicates, stray planes, unknown tensors and packed normalization tensors fail. Dense matrices can coexist separately with compressed matrices; mixed representations or coefficient dtypes within a fused group fail. Complete source catalogs/signatures include consumed duplicate planes. A quantization description without any packed matrix is rejected.

The CLI binds the registered compressed profile to its declared bit width and rejects compressed configuration under a dense profile. Catalog changes promote exactly the intended nine built-ins. Qwen2, Mistral, Granite and Gemma2 do not gain quantized support. Multimodal Gemma, newer generations, unimplemented quantization schemes, custom/per-layer descriptions, conflicting configuration aliases and unsupported semantic variants remain outside coverage. Catalog probing is separate from checkpoint validation and exact qualification.

**Family settings.** Llama preserves the accepted Llama3 RoPE bounds, tied/untied embeddings and existing bias restrictions. SmolLM3 preserves the pinned nonzero-mask-means-RoPE interpretation, omitted tied-embedding default, full-attention annotation checks, training metadata and optional biases. Gemma3 text preserves square-root embedding scaling in runtime configuration, normalization scale offset and upcast mode, Q/K norms, attention scaling, local/global windows and accepted global RoPE variants. Stored packed weights are not rescaled. Granite's scalar transforms remain a separate unsupported compressed contract.

**Workspace, publication and native contracts.** New family paths use the same bounded parameter tiles, source checks, output hashes, private staging cleanup and atomic no-replace publication as the reviewed Qwen3 route. The shared writer and native matrix/embedding/kernel implementations are unchanged. The native contracts accept the same 4/8-bit ScaleBias physical layouts and parameter shapes. Unit tests exercise four group sizes and nine source/target float pairs; real qualification here covers group 64 and BF16 only.

Six real artifacts matched the complete pinned external tensor/spec/config/tokenizer outputs: both bit widths of Llama3.2-1B, SmolLM3-3B and Gemma3-1B. Readiness followed by stdin EOF constructed the unchanged optimized Rust 1.98.1 worker with zero scoring calls. Exact qualification includes repository, dtype, source-manifest and artifact hashes plus the named build. It grants no scoring, generation, quality, current/future binary, other checkpoint/group/dtype or family-wide loading guarantee. F16/F32 conversion retains the runtime-blocked policy.

## Llama metadata transport provenance

The private recovery is accepted as explicit preparation of exact local content. The reviewer independently re-fetched unauthenticated public file metadata for the canonical official revision `9213176726f574b556790deb65791e0c5aa438b6` and public mirror revision `66da35cf6e4fcbdf48ecdba53c252715932d9a8b`. Both advertise the 189-byte `generation_config.json` Git blob `75ae08310d6d23df373ee2644b497192b3cce6d8`.

For both prepared Llama source manifests, the reviewer recomputed Git's blob hash and SHA-256 over the actual mirror bytes. Both match the canonical public blob and content hash `88effbb63300dbbc7390143fbbdd9d9fa50587b37e8bfd16c8c90d4970a74a36`. The source manifests retain the canonical FileSpec identity/revision and actual mirror-cache path. Separate mode-0600 receipts explicitly bind canonical and transport repository/revision, content hashes, reviewed evidence/script hashes and private manifest hash. Those bindings and mode-0600 manifests were checked independently.

The preparation wrapper allows only this exact official file/pin and the two reviewed primary checkpoints, verifies repository/file metadata before resolution, recomputes payload hashes, and refuses canonical-cache paths. Its code does not write an official-cache substitute or edit runtime routing, catalog, account, credentials or access forms. Ordinary remote resolution can still require official access. The new external reference resolver reads the same selected manifest path and checks its logical revision and complete SHA-256. The result is exact canonical content obtained through recorded public transport; it is not presented as a direct official download. The original gated preparation failures and initial timing batch remain preserved.

## Validation and performance assessment

The reviewer independently ran **872 targeted integrated tests**, covering the new frozen MLX family exports, shared MLX operator/numerical checks, CLI, registry, source integrity and exact qualification matching. A preceding independent negative/workspace run passed 146 cases. Initial review identified two stale dense rejection tests after valid MLX descriptions gained implementation; their narrow error-category corrections retain explicit failure before publication. No production blocker was found.

The actual pinned full-decoder reference generator covers 144 Llama/SmolLM3 exports across both bits, four groups and nine float pairs. Gemma tests cover 72 bit/group/dtype combinations plus thirteen complete frozen export/config cases. Twenty-one synthetic full CLI artifact comparisons passed exact tensor/spec/config/tokenizer parity with unchanged source/tooling and workspace at most 4,096 bytes. The reviewer inspected actual pinned loader/export use, independent registry/metadata resolution and aggregate acceptance flags. Low-level writers also pass with 96-byte workspace. Initial supplemental harness failures caused by requesting a CLI workspace below its 4,096-byte minimum are preserved separately; corrected harnesses passed without production changes.

The coordinator's final integrated suite records **1,625 passed and 39 skipped** with nineteen supplied private metadata references. Its actual public model-free run records **1,606 passed and 58 skipped**. Ruff/format, lock consistency, repository policy, seventeen CI-policy tests, actionlint with the existing disabled-job expression exception, and diff checks passed. The unchanged optimized Rust 1.98.1 build retains its previously recorded debug-info stripping warning. No new advisory-scan or hosted execution result is claimed.

The benchmark uses three repetitions per backend/tuple, serial fresh processes, warm selected files, alternating backend order, comparisons outside measurement and 10 ms process-tree RSS sampling. Setup/download/warming are excluded. Measured complete commands include source verification, metadata, output writing/hashes, final checks and receipts. The external reference performs weaker primary-source/receipt verification, so ratios describe those complete command contracts. No weight-only Llama surrogate is used in the definitive six-tuple report.

All six tuples satisfy workspace, sampled RSS, source/tooling and artifact-parity hard checks. All six median sampled RSS values are 90.7–96.6% lower than the external reference. The reviewer recomputed the timing ratios, RSS reductions and hypothesis flags from the final aggregate:

| Group-64 BF16 tuple | Stark median | Lalamo median | Raw latency disposition |
| --- | ---: | ---: | --- |
| Llama3.2-1B 4-bit | 1.845 s | 2.292 s | Passed; 1.24× ratio |
| Llama3.2-1B 8-bit | 2.233 s | 2.137 s | Failed; timing ranges overlap |
| SmolLM3-3B 4-bit | 3.979 s | 2.809 s | Failed |
| SmolLM3-3B 8-bit | 6.019 s | 3.544 s | Failed |
| Gemma3-1B 4-bit | 1.991 s | 2.444 s | Passed; 1.23× ratio |
| Gemma3-1B 8-bit | 2.655 s | 2.171 s | Failed |

The two passing tuples have separated observed timing ranges. These are descriptive warm-cache outcomes for the exact six tuples and host, with three repeats. RSS samples can miss peaks and count shared pages more than once; the workspace is not a whole-process memory ceiling. The four failed latency hypotheses remain open work. Root README, converter README and plan correctly expose all six rows, retain earlier measurements' source states, and separate conversion gains from inference speed/quality.

## Remaining gates

Complete Phase 4/G4 and replacement acceptance remain open for the remaining compressed/specialized/multimodal families, schemes, unqualified checkpoints/groups/dtypes, numerical/quality protocols for new quantization, strict process-memory enforcement, native compute/execution and broader complete-command latency parity. Existing models, operational metadata, replay manifests and native artifacts remain ignored and private; public documentation contains aggregate results and hashes. No Jev/JevK5 conversion, scoring or quality experiment was performed. No Rust/Metal format, scoring path, model distribution or inherited cloud/signing/release infrastructure changed.
