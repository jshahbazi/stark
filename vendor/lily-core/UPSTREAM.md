# Upstream provenance

- Project: [lily-qwen3.8-flash-next](https://github.com/fabiogreter/lily-qwen3.8-flash-next)
- Pinned revision: `e22d58fa7cc488493fb1b5d810a198bd29730aeb`
- Revision URL: <https://github.com/fabiogreter/lily-qwen3.8-flash-next/tree/e22d58fa7cc488493fb1b5d810a198bd29730aeb>
- Upstream metadata identifies Lily as Apache-2.0, Copyright 2026 Perplexity AI, Inc.; the modified lily-qwen3.8-flash-next distribution also attributes Copyright 2026 Fabio Greter. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Extraction and local changes

Stark extracts the minimum in-process Qwen4Exp model and native Metal 4 execution path into the `stark_lily_core` package, whose Rust library crate is `stark_lily_core`. The extraction is a modified subset, not an unmodified mirror. Stark removes server and command-line integration, dashboard and HTTP code, tools, session persistence, probes, and ablation controls. It supports resident experts only and excludes the disk-backed expert cache. The checkpoint-backed n-gram table may be paged read-only. The optional checkpoint MTP head can be loaded without changing its stored precision; vision execution remains excluded.

Stark adds `src/qwen4exp/host_spec.rs` and optional copies in the ordinary single-token decode graph. The bridge returns complete F32 vocabulary readouts to Stark's caller-owned exact sampler, without the upstream GPU sampler's 1,024-candidate cap or probability arithmetic. Draft distributions and acceptance stay request-local; head catch-up is disabled for ordinary non-MTP requests. Verification evaluates up to four positions in one layer-major graph with the ordinary decode's affine dot expressions, reductions, normalization and rounding boundaries. Recurrent state and pre-convolution inputs are retained for exact prefix rollback. The original serial verifier remains a reference. The engine adapter restores partially consumed groups when a stream stops early, without sampling during cleanup. Checkpoint quantization and the ordinary target's activation, state, cache and weight arithmetic are preserved. Numerical and performance qualification is recorded separately.

`src/qwen4exp/numerical_trace.rs` provides explicit in-memory stage comparisons in debug builds. Only aggregate mismatch counts, errors and named non-cryptographic fingerprints leave the scope. It has no file or environment interface, and release builds omit the capture fields and probes.

The official mixed-affine adapter also admits the checkpoint's declared
single-layer hybrid MTP head. Existing collision-checked aliases expose its
72 tensor planes without changing storage or quantization; the head shares
the trunk vocabulary projection. Its host verifier batches up to four target
rows using a separate official policy, including F32 ValueKey GDN captures
and exact prefix rollback. The adapter reports
`batched-affine-decode-exact-v1`; the ordinary serial verifier remains a
reference. Enabling MTP does not itself establish a speedup.

Host verification retains an owned flat F32 readout and lends complete rows
to the caller, eliminating the second per-row allocation and copy. The
buffer remains owned through sampling and prefix rollback. The original
Stark chunked finite guard in `src/finite_guard.rs` validates these and draft
readouts without changing any values; exact shape checks remain separate.
These host changes alter no GPU arithmetic or checkpoint precision.

Upstream environment-variable selection and diagnostics are disabled in the Stark integration. Precision is selected per operation: BF16 inputs and activations, F32 reductions, F32 logit buffers and F32 GDN recurrent state, plus explicitly selected Q8 attention K/V. Official head values are rounded to BF16 before widening into the F32 buffer. This policy description does not itself establish numerical equivalence or answer quality; source-bound component and model controls are recorded separately.

The upstream Metal shaders originate at the pinned revision. Stark modifies five of those files:

- `src/kernels/metal/hc.metal` adds cached-activation up/mix and row-axis verifier entry points reusing the original single-row bodies. These preserve weight staging, dot/reduction expressions and the BF16 activation boundary. Ordinary decode keeps its original dispatches.
- `src/kernels/metal/gdn.metal` keeps the gated output's explicit BF16 round in a thread-local value. Its verifier scan retains one F32 state column per thread across up to four positions, preserving the decode expression order and recording F32 states for rollback. The ordinary recurrent-state helper and norm reduction remain unchanged.
- `src/kernels/metal/attention.metal` adds row-axis query/key preparation copied from the decode bodies, preserving the norm reduction, BF16 rotary rounding and existing BF16/Q8 cache formats. Verifier attention retains the decode SDPA/QSA reductions.
- `src/kernels/metal/quant.metal` adds GDN Q4 two-row input projection with a convolution epilogue. It calls the existing packed dot helper, preserves the BF16 projection boundary and copies the original convolution expression, tap order and window update. Other layouts retain separate projection and convolution.
- `src/kernels/metal/moe.metal` adds a decode down/combine entry point whose row owner also performs the following hyperconnection injection. It copies the original dot and slot-ordered combine, retains the BF16 branch output, and applies the original F32 injection expression to that rounded branch. Decode orders both the branch output and residual streams before their next consumers. The original down/combine and injection kernels remain available for reference controls.

The other nine upstream shader files retain their pinned source bytes. `stark_decode_rows.metal` and `stark_moe_decode_rows.metal` add verifier entry points compiled after the unchanged generic projection and expert shader bodies. Batch row is the fast grid axis, exposing parallel work and weight-cache reuse while retaining each row's original arithmetic. The original HC and generic projection/convolution entry points remain available for reference controls. These patches do not change checkpoint quantization or the selected precision policy. Numerical and performance qualification is recorded separately in the [kernel study](../../docs/benchmarks/QWEN38_KERNELS_2026_10_29.md) and [injection fusion study](../../docs/benchmarks/QWEN38_FUSION_2026_10_29.md).

`stark_packed_rows.metal` extends the pinned Q4 packed-dot expression to independent short token rows while sharing raw weight loads. `stark_gdn_projection_rows.metal` adds the guarded verifier projection/convolution/scatter entry points. They preserve the 16-lane reduction, BF16 projection boundary and runtime convolution tap order, retain the original window, write its alternate, and capture projected inputs for rollback. Ordinary decode and unsupported layouts keep their existing routes. These original Stark extensions retain Apache-2.0 licensing; `quant.metal` is unchanged by this addition. The host verifier also batches up to four sparse queries through the existing QSA split kernels, using independent query scratch and per-query visibility; dense queries retain ordinary decode. Source-bound qualification is in the [MTP optimization study](../../docs/benchmarks/QWEN38_MTP_OPTIMIZATION_2026_10_30.md).

Stark adds `src/kernels/metal/stark_finite_readout.metal`, an original integer-bit validation kernel. After an ordinary GPU draw, it marks the output ID invalid if any F32 logit has exponent 255 (NaN or infinity). Stark's adapter rejects that sentinel before accepting a decision. This check covers both prefill and decode through the shared sampling helper and does not change the upstream model arithmetic. `src/kernels/metal/stark_gdn_step_reference.metal` is a test-only copy of the original gated-output wrapper, with a distinct entry-point name; it retains the shared-memory round trip for differential controls and is not included in production shader sources. Kernel controls adapted from upstream tests identify the pinned revision in their source headers.

Host patches also sanitize framework errors and retain submitted command memory until GPU completion feedback arrives after a fault. Missing completion feedback keeps fault cleanup blocked; it cannot safely release live GPU memory. Unused capture, profiling and standalone vision helpers are removed from the extracted Rust modules.

Stark adds a canonical Uzu storage admission path in `src/safetensors/uzu.rs`
and refactors Qwen weight/model loading so native-source and typed Uzu inputs
share the existing specialized kernels. The host's
`Qwen38LanguageModelConfig` contains explicit text architecture fields and
logical tensor bindings. Admission requires one regular local
`model.safetensors`, canonical MLXSpec matrices, bounded headers, checked byte
spans, a bijective tensor inventory and valid layout/dtype/shape combinations.
U32 packed codes are exposed from identical U8 storage bytes; BF16 group-major
linear/expert parameter planes are inverse-transposed into owned logical
buffers, with zero padding checked. Embedding row-group planes retain their
paged access layout. Direct BF16/F32 and I64/U64 controls preserve their bits;
optional PLE hash buffers must match the typed configuration as exact raw
64-bit words. The converted folder supplies the configuration, tokenizer and
weights, with no fallback to an original checkpoint folder. The typed text
artifact omits vision and MTP. These are host loading and storage changes;
their numerical and performance validation is recorded separately in the
[Uzu-format study](../../docs/benchmarks/QWEN38_UZU_FORMAT_2026_10_29.md).

Stark adds an explicit, disabled-by-default Metal 4 stage profiler in `src/metal_stage_profile.rs`. Fixed phase markers use a bounded per-pass timestamp heap, retain the existing concurrent dispatch and dependency barriers, and require one command buffer per profiled decode. The heap remains owned through verified completion and fault draining. Resolution checks entry sizes, timestamp order, device-reported frequency and agreement with independent command feedback. Only aggregate phase durations leave the profiler; no model values or request payloads are captured. Precise timestamps can still change scheduling, so instrumented rates are diagnostic and separate from ordinary throughput runs. Upstream diagnostic environment controls remain disabled.

`MslMathMode::Safe` in `src/metal.rs` is an explicit per-library and per-pipeline compile policy matching the pinned MLX `64ea011cb65f14d9ce2737e60db9a4ae91ed7441` `CompileOptions::Safe` behavior. It requests `MTLMathMode::Safe` when the selector is available and uses the legacy `fastMathEnabled = false` equivalent otherwise. Pipeline cache keys include the math mode. Existing `pipeline` and `compile_library` calls retain `PlatformDefault`; this API does not change their compiler policy. Safe mode is selected only by wrappers whose source and rounding policy require it.
The same bounded profiler covers host-sampled MTP verification, with separate MoE setup, gate/up/shared-down and down/combine categories. The adapter records fixed aggregate draft, verification, target-sampling and finish scopes. The terminal verifier timestamp precedes the completion signal so profiling retains one command buffer. Disabled profiling adds no timestamp or counter-heap work; failures report fixed diagnostic categories without framework messages or model data.

Every successful pass wait also awaits all asynchronous completion feedback before accepting outputs, then checks the context's sticky fault. A signaled GPU fence alone is insufficient: framework error feedback can arrive afterward. Command resources remain held until these checks succeed, and failed checks retain the fault-safe cleanup drain.

## Official MLX-Serve affine checkpoint adapter

Stark adds direct loading for the external-table affine Qwen checkpoint dialect
used by MLX-Serve 26.10.3. Tensor-name aliases retain each source file, byte
interval, dtype and shape. The external `ngram_table.bin` is mapped as one
logical PLE shard. Q4/Q8 code, scale and bias planes remain unchanged; the small
BF16 HC injection and shared-expert gate matrices use direct BF16 projection
dispatch with F32 accumulation. This adapter does not requantize weights.

The PLE hash-control derivation in `src/qwen4exp/config.rs` is adapted from
[`src/qwen4_exp.zig`](https://github.com/ddalcu/mlx-serve/blob/046b647686419df06c6c7132fb1d15d76185c3e5/src/qwen4_exp.zig)
at MLX-Serve revision `046b647686419df06c6c7132fb1d15d76185c3e5`
(tag `v26.10.3`), Copyright 2026 David Dalcu, MIT licensed. Its copyright and
permission notice are retained in [LICENSE.mlx-serve](LICENSE.mlx-serve).
The external-pack folded-normalization policy follows the same revision's
[`resolveAndFoldQwen4Norms`](https://github.com/ddalcu/mlx-serve/blob/046b647686419df06c6c7132fb1d15d76185c3e5/src/model.zig).
GDN's plain-gain normalization remains separate.

The original Lily Metal shader bytes remain unchanged by this adapter.
Folded attention/indexer normalization uses distinct compiled variants of the
existing source; the original entry points retain Lily's delta-weight gains.
`src/kernels/metal/official_gate.metal` is a new Stark implementation for the
direct BF16 gates. Lily's affine HC, GDN, MoE and projection paths remain
available under their existing precision and geometry admission checks.

## Validation boundary

This document records source lineage, implementation boundaries and licensing;
it does not publish dated run results. Exact component controls, whole-model
numerical controls and sampled throughput are separate evidence levels and are
recorded in the dated benchmark/run record. Do not infer full-model quality
equivalence or a performance improvement from source correspondence alone.


## Bounded official-checkpoint MoE numerical policy

The bounded path in `src/kernels/metal/official_moe.metal` and `moe.rs`
adapts MLX-Serve `046b647686419df06c6c7132fb1d15d76185c3e5`'s served fused
router and affine4 decode contracts from transformer.zig and moe_affine4.zig.
The MIT attribution is retained in NOTICE and LICENSE.mlx-serve. It uses
distinct shader entry points and leaves the original moe.metal unchanged.

The strictly admitted external checkpoint policy selects BF16 decode router
logits/shared gate, BF16 full-softmax probabilities and sequential selected
probability normalization. F32 scratch scores hold exactly rounded BF16
values. The served single-row affine4 shape (512 experts, top10, hidden2560,
intermediate640, group64) gets separate x16 dot geometry, BF16 table-based
SwiGLU, and score-weighted F32 down accumulation before the BF16 output.
The shared epilogue uses the BF16 primitive sigmoid, product and addition;
the official HC write remains a separate qualified operation.

Prefill uses the existing grouped expert projections instead of unqualified
legacy small-m fusions, BF16 SwiGLU boundaries, and eight BF16 partials for
the score-weighted combine. This follows the served fused prefill reduction
and the canonical K10/H2560 col_reduce_small fallback. Shared/quantized GEMM
reduction parity is not established by this operation-rounding patch.

The finite-domain sigmoid table is generated once outside generation. Its
primitive construction and fused affine4 paths have dedicated source-bound
controls. This source description does not claim parity outside those controls,
full-model quality equivalence, or throughput improvement; those results are
recorded separately.


## Bounded official affine64 KV cache

The new, explicit `Q8Affine64` cache's encode algorithm is adapted from
MLX `mlx/backend/metal/kernels/quantized.h`, `affine_quantize`, at
`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`, the MLX gitlink pinned by
MLX-Serve `046b647686419df06c6c7132fb1d15d76185c3e5`. The MIT notice is
retained in `LICENSE.mlx`. The Metal scheduling and metadata layout are
adapted for Stark: two SIMD groups hold each 64-value group and cache metadata
interleaves BF16 scale/bias pairs. Stored checkpoint weights are unchanged.

The pinned MLX C-API oracle `matched-mlx-affine8-oracle-v1.json`, SHA256
`40655a784a2576e74f8fe2b206146b229bec40f1b8adb573ed972fa2fc0ddec2`,
establishes one final BF16 round after F32 code-times-stored-scale plus stored
bias. Rounding the product to BF16 first failed that reference. The official
cache route is selected only by its explicit policy and supported affine
layout. Existing symmetric Q8 and BF16 paths retain their source and entrypoints.
The encoding source and its exact stock controls do not establish full-model
numerical or quality equivalence; those controls are documented separately.

## Bounded official ordinary BF16 RMS and fused normalization

The standalone official_norm.metal and shared fused official_norm_common.metal
adapt MLX rms_norm.metal::rms_single_row at
64ea011cb65f14d9ce2737e60db9a4ae91ed7441. Apple copyright and the MIT license
are retained in the sources and LICENSE.mlx.

The actual pinned stock-library GPU C API RMS report
matched-mlx-fast-rms-oracle-v2.json, SHA256
a76b2034b3f4c603422c109b654cc6535e90db0482a7704ad3177b2724ec8b25,
establishes normalized-activation BF16 rounding before BF16 gain, then another
BF16 round. F32 gain promotes the output to F32; these new BF16-only wrappers
refuse it. The single-row tree uses four consecutive inputs per reduction
worker, zero-padded F32 partials and precise reciprocal square root.

New explicit-policy wrappers cover standalone attention/MTP embedding norms,
decode Q/K normalization, and all four QSA query/block-key position variants.
Old shader source files and false-policy wrappers remain unchanged. These
wrappers are present in the current source with explicit dtype and output
admission. Their stock reference semantics do not establish whole-model output
equivalence or throughput; qualification is recorded separately.


## Bounded official attention output gate

The separate official attention gate wrapper reuses the already-qualified
MoE scratch BF16 sigmoid table. Its table-lookup/product follows
MLX-Serve 046b647686419df06c6c7132fb1d15d76185c3e5,
transformer.zig STRIDED_SIGMOID_GATE_MUL_SOURCE and gatedAttnTail.
Copyright (c) 2026 David Dalcu, MIT; see LICENSE.mlx-serve.
The false-policy native shader is unchanged. The official wrapper is selected
only under its explicit policy and admitted geometry. Source correspondence
does not substitute for separate exact component controls or establish
full-model output equivalence.


## Bounded official HC read and WR1 pair

The separate official_hc_stock.metal ports ND_SOURCE and the actual served
vectorized U2V_SOURCE from MLX-Serve
046b647686419df06c6c7132fb1d15d76185c3e5, src/hc_decode2.zig SHA256
021ae2790bbad52c0540092685892587277412afc96fcc0b6156dfd40209dab3.
The original arithmetic bodies are copied verbatim; the native bindings fix
H2560/G4/R320/Q8/group64/rows1/WR0, with separate injection on/off entries.
David Dalcu's copyright and MIT license remain in the source and
`LICENSE.mlx-serve`. Fixed one-row descriptors own their derived projection
buffers; unsupported layouts use the retained HC paths. The current source also
contains a WR1 pending-write pair: it is selected only when both adjacent
one-row reads are admitted, both carry injection, injection is enabled, and
pair scratch is available. It completes back into canonical hyper storage
before downstream consumers. The separate WR0 path remains the fallback.
WR1 source provenance and bounded selection do not imply exact full-model or
quality parity.

The actual pinned-library C API oracle stock-hc-decode2-oracle-v2.json,
SHA256 0053013d874c0082e2a4bfe48770f1a1176a006c9d296f67587af92b5ace818b,
ran both nontrivial fixture seeds with injection on/off and exact repeats.
Native synthetic controls bind the same input hashes and F32 partial/BF16
stage hashes. Those references do not establish native compiler, full-model
or performance equivalence until their separate runs pass.

Only admitted official one-row reads select this descriptor. Each loaded
model owns its load-time folded BF16 down metadata and scaled/transposed
dense BF16 injection copies; original checkpoint buffers and bit widths are
unchanged. A fixed one-row F32 parts buffer is scratch, not a context-sized
allocation. The source changes native F32 projection/reduction association
to match stock, so identity with the previous skinny projection is not claimed.

U2V returns a finished BF16 injection gate. The graph uses an explicit
RawLogits/PrecomputedGate kind governed by the same read eligibility and read-enabled predicate,
including normal, batched and MTP writes. HC-ablated skipped decode reads
retain raw-logit interpretation, with a pure predicate regression that does
not modify the process-global ablation environment. Finished gates only enter the
BF16 product/residual-add write, preventing a second sigmoid. Larger row
counts and unsupported descriptors retain their existing read/write paths.


## Bounded official TB12 GDN prefill recurrence

The separate official_gdn_pipelined.metal retains the Apache-2.0 attribution
from oMLX jundot/omlx revision 730bd6a, PR #4006, through MLX-Serve
046b647686419df06c6c7132fb1d15d76185c3e5, src/kernels/gdn_pipelined.metal.
The 609-line stock source SHA256 is
4de16729aa7614a55215acfa14c7df69094ba52b4018b138c1c5ff6b4333d86a.
The fixed batch1/Hk16/Hv48/D128 adapter changes only Q/K, V and Y row
strides in the recurrence body; reversing those substitutions reproduces
the stock body exactly. Its native entrypoint is unique for the pipeline
cache's function-name/MSL-version/math-mode key.

Only official ValueKey prefill with at least 64 rows, no per-position capture
and aligned vector bases selects it. It reuses existing BF16 input/QK and F32 gate staging, keeps
persistent F32 state [H,V,K], allocates no global scratch, and requires SIMD
width 32. Short, captured, unaligned and retained KeyValue/Lily paths preserve their
existing reference dispatches.
Safe MSL3.1 is explicit and local to the new kernel. Stored weight bits,
checkpoint buffers, quantization and speculation settings do not change.

The independent matched-mlx-gdn-pipelined-oracle-v1.py executes the exact
stock C API shader at seven full-block/tail/route-boundary sizes and emits
raw BF16 Y and full F32 state hashes from identical staged synthetic inputs.
Separate sparse65 exact-stock prework/recurrence/norm controls record Q/K,
V, gates and convolution history; they do not qualify dense prework or norm
reduction arithmetic. Native prework and norm policy remain unchanged.
The selected source path is bounded by the route predicate and runtime shape
checks. Its source lineage and synthetic stock fixtures are not full-model or
throughput qualification; those results are recorded separately with their
controller and receipt identities.


## Bounded contiguous official prefill Q/K path

The separate official_gdn_prefill_qk.metal adapts only the Q/K normalization
fragment from MLX-Serve 046b647686419df06c6c7132fb1d15d76185c3e5,
transformer.zig GDN_PREWORK_SOURCE, MIT, Copyright (c) 2026 David Dalcu.
It uses four contiguous BF16 values per lane, the stock F32 accumulation/SIMD
reduction and precise reciprocal square root, then the two BF16 rounds.
It consumes existing activated QKV and writes existing interleaved Q/K
staging. Scoped Safe MSL3.1 and a unique entrypoint isolate compiler policy.

Only the admitted official ValueKey prefill route selects it: at least 64
consecutive rows, no mid-state capture and aligned original buffers. Other
layouts, lengths and capture modes retain their existing Q/K helper.
Convolution, gates, norm, TB12 source, persistent F32 state and stored weight
quantization do not change. This source path and its synthetic controls do not
establish dense full-chain or full-model equivalence. Component and end-to-end
numerical results are maintained separately.


## Bounded official Qwen3.8 decode policies

The ValueKey implementation in `src/kernels/metal/gdn.metal` adapts GDN decode
`K1_SOURCE` from MLX-Serve `046b647686419df06c6c7132fb1d15d76185c3e5`,
`src/gdn_decode.zig`; Copyright (c) 2026 David Dalcu, MIT. Its test-only
`official_gdn_k1_helper.metal` contains a separately named helper for exhaustive
activation-boundary controls. The permission is included in
`LICENSE.mlx-serve`. In `src/kernels/gdn.rs`, these stock-order BF16 activation
boundaries are selected only for the official ValueKey layout with
Hk16/Hv48/D128, BF16 gates and QKV, widened F32 A-log and F32
`[48,V128,K128]` recurrent state. Other layouts retain the existing path.
Convolution, recurrence and norm/gate are separate stages with separate
entrypoints and a local Safe MSL3.1 policy. Source lineage does not establish
complete-model equivalence.

`src/kernels/metal/official_head.metal` supplies a stock QMV fast path adapted from MLX
`64ea011cb65f14d9ce2737e60db9a4ae91ed7441`,
`mlx/backend/metal/kernels/quantized.h`; Copyright (c) 2023-2024 Apple Inc.,
MIT. The fast route is limited to official BF16 activations, F32 output storage,
Q4/group64 weights, K=2560, N divisible by eight and signed-offset-safe code
spans and a 32-wide execution pipeline. Other valid head geometries retain generic GEMV plus the existing BF16
value-rounding pass. The fast path is compiled as Safe MSL4.0. No checkpoint
weight data is changed. Stock source expressions and exact component controls
do not establish full-vocabulary or
full-model parity.

`src/kernels/metal/official_q8_gdn_decode.metal` adapts the MLX affine QMV
fast-loop expressions from the same pinned `quantized.h` revision and license.
The GDN decode wrapper in `official_q8_projection.rs` is selected only for the
explicit official rounding policy, Q8/group64 rank-2 weights, BF16 activation,
output and affine metadata, and `(N,K)=(16480,2560)` or `(2560,6144)`. It also
requires a 32-wide execution pipeline and is compiled as Safe MSL4.0.
Non-admitted shapes and widths delegate to the original GEMV; pipeline
compilation errors remain errors. Prefill callers are unchanged. The exact
official verifier exposes the helper's existing vector-grid axis for one
to four rows; each row retains the same stock loop and BF16 boundary.
Stock hashes and native qualification are recorded separately; this provenance
is not a model result.

## Official short-row MTP verification

The official verifier adds independent short-row execution to the policies
above without changing checkpoint tensors. `official_moe_rows.metal` copies
the qualified MLX-Serve router/expert arithmetic from the pinned local
`official_moe.metal` port and adds token-row addressing. It retains the
expert-order down accumulation, BF16 router/activation boundaries and the
folded shared gate. The original shader bytes remain unchanged. Copyright
(c) 2026 David Dalcu and the MIT license in `LICENSE.mlx-serve` apply.

`official_hc_stock_rows.metal` adapts the pinned HC U2V body with a compile-time
row count and shared up-weight loads. The original ND body already has a
row-grid dimension. Per-row partial storage is bounded to four rows. A
separate finished-gate write preserves the two BF16 product/add boundaries.
The original one-row kernels remain selected for ordinary decode. The new
file retains the MLX-Serve revision, copyright and MIT attribution.

The stock-order Q4 vocabulary-head helper in `official_head.metal` shares
packed weight loads across two to four rows while preserving each row's
affine dot and SIMD sum. The existing MLX revision and MIT attribution above
apply. Non-admitted head formats execute ordinary projection per row.

The mixed-4-8bit checkpoint's actual vocabulary projection is Q8/group64;
its global Q4 default does not describe the `head` storage group. The new
`official_head_q8_rows.metal` adapts Lily's `quant.metal` `GEMV_Q8_BODY`,
retaining `dot_word_q8`, per-lane accumulation, SIMD sum and PlatformDefault
compiler policy. It shares packed weights across two to four verifier inputs
and fuses the existing integer round-to-nearest-even BF16 value store into
the F32 output. Ordinary one-row decode remains unchanged. Lily's Apache-2.0
notice and the MLX-derived dot attribution in `NOTICE` remain applicable.
Its complete-readout component controls include actual full-vocabulary
geometry, aligned nonzero offsets and output guards.

The ValueKey verification additions in `gdn.metal` retain the existing
MLX-Serve-derived K1 convolution, recurrence and norm/gate expressions. They
scan at most four positions with F32 physical state captures after each
rollback prefix, retaining the ordinary BF16 activation boundaries. Each
path has component comparisons against ordinary decode. Full-model
readout/rollback and sampled performance records are maintained separately.
