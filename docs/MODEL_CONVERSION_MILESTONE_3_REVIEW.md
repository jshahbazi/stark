# Independent conversion milestone 3 technical review

Date: 2026-10-11. Baseline: merged `origin/main` at `41db29b7`. The reviewer authored no production implementation or qualification coordinator. This review covers the first compressed-format packet and its source-bound evidence in [milestone 3](MODEL_CONVERSION_MILESTONE_3.md).

## Disposition

**Accept the scoped Qwen3 MLX affine conversion, selected BF16 load-only, bounded-workspace, and measured performance promotion.** No unresolved blocking finding remains. Twelve additional pinned Qwen3 4/8-bit profiles have an independent conversion route. Two exact 0.6B artifacts have complete external parity and named-build native load evidence. The package now implements 45 of 110 pinned built-ins plus four separately counted extensions; 65 built-ins remain unimplemented.

This disposition accepts the recorded compressed G4 packet. The remaining compressed families and schemes, broader checkpoint/group/dtype qualification, execution, quality, and complete Lalamo replacement gates remain open. Earlier dense latency failures and historical prototype results retain their original source-bound status.

## Reviewed source and evidence binding

| Artifact | SHA-256 |
| --- | --- |
| final converter sources | `42a883a195b095b5812a0bbb5502b85ef8efa5840b4181f6d347fa11d30fea0c` |
| converter lock | `d230bd2dc0681a13a081876eb7677bd22b7d1f029bf2d046cc0bf2e50156e658` |
| catalog | `4db2607b2a5ad06766d0f3667d99706c413e5649dffdaf631775f638c1d424c1` |
| thirteen exact native qualifications | `61b8b1ae7fc5abadbe7eb1e469ac24b3cf6ca175b4d1f7b85fc25c2b10b8501d` |
| two-checkpoint official benchmark | `03b7bab9fd26503dcfae901ae7d3fb399aa1a2a627c2d2bea0fae1674bcf319c` |
| qualification summary | `739f5c2b090597df67906a184a5ab743a37e9590bf31268681d4a5c7d1301e06` |
| 288 synthetic matrix/export/numerical references | `6e90f3262ce84f7e0f791dd250fc05078929f1a0c26b7dfaf196b01d50dc5d15` |
| twelve corrected complete synthetic comparisons | `d4efc149a4f62831d40ca008337eada086b6295e3d3c86ef4dac5c6df3f1aeb9` |

The reviewer recomputed current source/lock and aggregate hashes, verified both benchmark tooling bindings, inspected acceptance flags, and matched all six final-source timed candidate receipts against the exact source/weights/config/tokenizer qualification ledger. All eleven prior ledger records are preserved unchanged. Both new native records report successful loading with zero scoring calls on worker `61bb81a83fa368b7c8f647469ac468048cf6868d973ed74e7d9d62633bfba96d`.

Initial untimed qualifications retain converter source hash `7e7a458262cd43224f990b78d2ed9c7352bad0904dda25bfafe74e5422a89593`. Adding the two qualification records and refining packed-operation validation produced the final source hash. Six fresh final-source candidate conversions passed exact stored-artifact matching and complete external comparison. Earlier evidence was not silently rebound to the final source.

Reference semantics were checked directly against the clean Lalamo checkout at `831accb8252a6b6606c378048caeb44178f1a050`: its Hugging Face packed loader, byte packing, MLX matrix export/decompression, affine parameter layout, and Qwen3 configuration/profile. Existing MIT attribution and `LICENSE.lalamo` remain intact. Version 0.4.0 changes the companion-package version only; dependency versions are unchanged. The optional reference/export tools remain outside the converter runtime.

## Technical assessment

**Packed matrices.** The source weight words are I32/U32 little-endian values with low-bit-first codes. For accepted 4/8-bit checkpoints, the raw word bytes already equal the pinned loader's unpack-then-byte-pack result. The converter copies that payload and exports U8 metadata with four times the word width. Signed words retain their high bits. Full tensor geometry, logical dimensions, packing/group divisibility, both affine plane shapes, float dtypes, and declared quantization descriptions are checked before writing.

Q/K/V fuse in Q, K, V order; MLP matrices fuse in up, gate order. Affine planes follow those same source rows across shards. Input/tied embeddings retain physical `[vocab, packed_model_dim]` weights and untransposed `[vocab, groups]` planes. Projections and untied readout export planes as `[groups, round_up(output_rows, 4)]`, padding after the complete fused row count. Optional linear bias vectors remain separate from MLX affine bias planes. Dense matrices may coexist outside a fused group; mixed dense/packed fusion and mixed plane/fusion dtypes fail explicitly.

**Workspace and publication.** Parameter export casts and transposes bounded tiles without unpacking a complete weight matrix. The budget reserves the input tile and row read, cast output/scratch, half-cast NaN mask, transpose output, and numeric-zero padding. Arrays are released between tiles. The existing completed-file hash path handles seeks; sequential packed copies preserve inline hashing. New operation validation checks each source byte count, dtype, shape, output padding, and forbidden scalar fields before output creation.

Complete original source catalogs and signatures include accepted duplicate tied-head planes. Every tied duplicate plane must match its input-embedding counterpart. The existing source resolver, final complete content verification, tooling checks, private staging cleanup, and atomic no-replace publication remain in effect. The CLI's new end-to-end regressions exercise external metadata pinning/offline replay, catalog bit mismatches, dense-profile rejection of packed configuration, and parameter-writer failure cleanup. No alternate framework or delegated conversion route was added.

**Native and capability boundaries.** Owning native matrix/embedding contracts accept 4/8-bit ScaleBias, require matching physical layout/plane dtype, and use the same parameter shapes. Existing Metal policies support the initial 16/32/64/128 group choices through their selected tile routes; actual compute dispatch remains untested by load readiness. The two recorded real artifacts use group 128 and BF16. Their exact identities and hashes qualify loading only for the named unchanged Rust 1.98.1 release/default-plus-optimized build.

Catalog probing does not validate source content or grant exact qualification. Publication grants a recorded qualification only when repository, dtype, source manifest, and every artifact hash match. Other compressed artifacts remain implemented-unqualified. F16/F32 conversion remains available with the existing runtime-blocked policy. Unsupported 1-bit, AWQ/GPTQ, per-layer/custom quantization, hybrid/state-space, and other packed families receive no fallback.

## Validation and measured outcomes

The reviewer independently ran **581 targeted integrated tests**, covering MLX operators/numerics, CLI, registry, source integrity, and exact qualification matching. The initial review caught a regression test using the wrong Qwen3.5 catalog owner; that test was corrected and the integrated rerun passed. No production blocker was found.

The coordinator records **1,310 passed and 36 skipped** with thirteen supplied real metadata references; the public model-free run has 1,297 passed and 49 unsupplied-reference skips. Ruff formatting/checks, lock consistency, repository policy, and diff checks passed. The unchanged optimized Rust 1.98.1 build succeeded with the existing debug-info stripping warning.

Frozen external matrix references cover 288 combinations of bit width, group size, source/target dtype, physical layout, and single/fused rows. All 4/8-bit code values, signed-word payloads, independent row/group affine coefficients, global output padding, and cross-shard fusion are observable. Integer payload/spec and binary-exact synthetic dequantization tolerances are zero. The reviewer inspected the generator's use of the actual pinned loader and exporter/decompressor. Twelve complete synthetic artifact comparisons cover tied/untied and dense-embedding variants, optional biases, all nine float dtype pairs, and source/workspace stability. Their final source hash is unchanged across the run. The earlier failed synthetic run is retained separately; correcting fixture registry resolution produced the accepted run without a production change.

Two real immutable `Qwen/Qwen3-0.6B-MLX-4bit` and `Qwen/Qwen3-0.6B-MLX-8bit` artifacts each matched all 452 output tensors, tensor descriptors/specs, parsed configuration, and tokenizer against the pinned external converter. Canonical content parity permits serialization/header formatting differences; exact candidate file hashes remain separately recorded. Both readiness/EOF checks performed zero scoring calls.

The benchmark uses three repeats per backend/checkpoint, serial fresh direct processes, warm selected source files, alternating backend order, parity outside timing, and 10 ms tree-RSS samples. All twelve timed artifact comparisons, six candidate exact qualification matches, workspace/RSS hard checks, and both latency/memory hypotheses passed. Complete command medians were **1.011 s versus 1.796 s** for 4-bit (1.78× ratio), and **1.506 s versus 1.818 s** for 8-bit (1.21× ratio). Observed timing ranges are separated for both tuples. Candidate median sampled tree RSS was approximately 188 MiB for each, approximately 93% below the external reference.

The measured commands include source verification, metadata, output hashes, final checks, and receipts; external Lalamo performs weaker verification/receipt work. These are complete-command outcomes for the two recorded tuples and protocol. RSS sampling may miss peaks and does not enforce a process memory ceiling. Neither ratio establishes family-wide latency, inference speed, or model-quality improvement.

## Remaining gates

Full Phase 4/G4 acceptance remains open for other compressed families/schemes, broader variants/groups/dtypes, non-binary affine numerical protocols, new quantization, native compute/quality, and strict process-memory enforcement. Sixty-five built-in adapters and complete replacement accounting remain outstanding. Earlier dense performance outcomes are unchanged.

No Rust/Metal format, kernel, inference engine, or scoring-path change is present. No Jev/JevK5 conversion, inference, or quality experiment was performed. Real models, operational metadata, private manifests, and runtime evidence remain ignored; public artifacts contain aggregate outcomes and hashes. Prior source-bound results and provenance remain preserved. Inherited cloud/signing/release infrastructure stays disabled.
