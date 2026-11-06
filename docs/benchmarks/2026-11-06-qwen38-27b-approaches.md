# Dense Qwen27 approaches and decisions

This ledger closes the native `Jundot/Qwen3.8-27B-oQ4e-mtp` investigation at
**51.483 useful decode tokens/s median** on the M5 Max with 128 GiB memory.
All six 768-token responses passed host screening on their first attempt.
The 70.8 tokens/s objective remains unmet. Further optimization is paused.

The [initial implementation report](2026-11-06-qwen38-27b-oq4e-mtp.md) and
[packed-weight follow-up](2026-11-06-qwen38-27b-packed-followup.md) retain their
separate build identities, numerical checks and timing controls. This ledger
adds the decisions and unsuccessful attempts without rewriting either run's
evidence. Its implementation scope is commits `809da572` and `a4579906`.
The final source, artifact and executable identities remain those in the
follow-up report; this documentation does not constitute a new inference run.

## Outcome definitions

- **Kept:** present in the shipped branch. A structural improvement or
  correctness repair does not by itself establish an isolated speedup.
- **Optional:** implemented and checked, with the original policy still used
  by default because whole-model performance did not justify enabling it.
- **Removed:** implemented or prototyped, then excluded from the shipped path.
- **Investigated only:** inspected or measured externally; no native production
  implementation or matched whole-model gain is claimed.

## Native execution and numerical policy

| Approach | Outcome | Evidence and reason |
| --- | --- | --- |
| Native dense graph with the complete declared one-layer MTP predictor | Kept | Converts and executes the pinned dense checkpoint in Rust/Metal. The 64-layer target has 48 DeltaNet and 16 full-attention layers. It requires its own admission and validation; Flash-Next results do not qualify it. |
| Lossless widening of five/six-bit and mixed-width matrices to eight-bit storage | Superseded for eligible matrices | Established a reference implementation without changing source integers or affine coefficients, but expanded the weight file to 21,197,528,944 bytes. Input-output embeddings and other layouts outside packed admission retain their lossless widened representation. |
| Closed-form batched DeltaNet update for speculative verification | Removed from this verification route | One of three bounded MLX/native continuations diverged. Teacher forcing showed drift after the first row even with all proposals accepted, so rejection rollback was not the cause. |
| Serial-equivalent TreeVerify/StateAdvance arithmetic for flat verification rows two through four | Kept | Repaired that drift while retaining batched projections. The failing case then matched all vocabulary logits at all 32 audited positions. The three bounded reference continuations matched. This does not establish equivalence to every ordinary full-accept DeltaNet update. |
| Fused M5 short-chain DeltaNet verifier | Kept | Fuses Gram, identity solve, output and separate state advance while preserving the checked per-row arithmetic and speculative-state ownership. The admitted head geometry passed all-vocabulary audits with zero error; other geometries retain the fallback. Its isolated whole-model speed contribution was not established by a matched screened A/B. |
| oMLX post-normalization proposal policy | Kept | Apply the target's final norm to captured target residuals; feed the predictor's normalized output into subsequent draft steps. This matches the inspected pinned oMLX policy. Head replay established bounded seam agreement, with remaining cross-engine BF16 matmul differences documented in the initial report. Exact rejection continues to use the unchanged target. |
| Readout/state-copy and batched-preparation iterations | Kept in the final implementation; individual speed attribution unresolved | Intermediate `copy-readout`, `batched-prep` and `postnorm` workers have private aggregate receipts. Their unscreened six-response medians were 32.052, 36.330 and 44.508 tok/s; these successive builds are not matched causal comparisons. The final path reads contiguous BF16 vocabulary output once, owns captured hidden rows on device, shares batched projections/preparation, and reads only the final prefill vocabulary row. Aggregate receipts alone do not reconstruct each intermediate patch's isolated effect. |
| Exact prefix reuse for the dense recurrent target and predictor | Kept | Restores the previous request's chunk-aligned state on a 1,024-token grid, keyed by model instance, scope, context and exact selected draft depth. Final original/three-bit/four-bit head artifacts each passed 40 comparisons and six invalidation controls. Initial prefix timing observations are separate, unscreened latency evidence. |

## Quantized matrices and kernel experiments

| Approach | Outcome | Evidence and reason |
| --- | --- | --- |
| Native packed affine five/six-bit matrices | Kept | Preserves source codes and BF16 affine coefficients for admitted group-64 output-input matrices. Together with row partitions, reduces the weight file to 17,078,038,512 bytes, saving 4,119,490,432 bytes. The balanced, unscreened packing-only A/B measured 45.250 to 48.957 tok/s (+8.19%), with matching completion hashes. |
| Heterogeneous fused projections retained as contiguous row partitions | Kept after numerical repair | The first restored-width candidate changed four-bit lane reductions and original fused-output split-K planning and failed the numerical gate. The repair retains widened-eight-bit arithmetic and the original fused output dimension while reading actual packed codes. Final widened/packed and scalar/four-row comparisons matched all 24,583,680 logits exactly. |
| Smaller affine GEMV threadgroups, with unchanged reduction arithmetic | Removed as a general policy | The synthetic component sweep found only small W4 changes and no useful general W8 improvement. It did not justify a broader selector change. This is distinct from the retained narrow row-tile exception below. |
| Two input rows per dispatch for W8 FFN-down at four verification rows | Kept, narrowly gated | For `M=4, N=5120, K=17408`, the component sweep measured about 226.76 microseconds with input-row tile four versus 180.55 with tile two, retaining the eight-simdgroup, 32-lane reduction geometry. The selector keeps this exact-shape exception; other admitted shapes retain their selected row tile. This is component evidence, not an isolated end-to-end speedup. |
| MLX-style eight-lane K reduction (`affine-k8`) | Removed | Its new scalar and batched paths agreed internally, but the reduction changed old-versus-new numerical results and ran slower on most tested actual matrix shapes. Small wins on some W4 FFN-down cases did not justify promoting the broader numerical and dispatch policy. This is not the unrelated split-K-eight trial from earlier Flash-Next work. |
| Small-batch simdgroup matrix-MMA (`sg8`) prototype | No native port | Exact integer operands, CPU-F64 reference checks and scalar/batch gates passed. Observations did not demonstrate the predeclared 15–20% port threshold, and active W8 cases were slower. Prototype synchronized wall timing and historical native GPU timestamps use different timers/datasets, so contextual percentages are not a matched native speedup. |
| Two-aligned-word unpack for packed five/six-bit codes | Removed | It did not establish a reliable measured gain under the observed host drift. The evidence does not isolate a hardware-level cause or establish that this unpack scheme is always slower. Private source and receipts remain available. |

The narrow tile-two exception is selected in
`crates/stark-engine/src/backends/metal/kernel/matmul/gemv/policy/quantized.rs`.
The other kernel candidates above do not remain as unused production selectors.
The source widths, precision policies and exactness controls were maintained
separately from timing results.

## Draft readout, state maintenance and scheduling

| Approach | Outcome | Evidence and reason |
| --- | --- | --- |
| Three/four-bit coarse draft head, GPU shortlist of 64 and exact original-head rescoring | Optional | The MLX component experiment measured 2.016 ms for the original head versus 1.228 ms for three-bit plus rescoring and 1.493 ms for four-bit plus rescoring. Native ranking/rescoring and both artifact-bound prefix paths passed. Whole-model screening did not demonstrate a reliable gain: a paired original/three-bit screen measured 46.808/46.146 tok/s pooled medians, with only two complete host-eligible pairs. The full original head remains the default. |
| Initial coarse-head trial using F32 scores with a BF16-only selector | Invalid trial; repaired | The storage/type mismatch invalidated that trial's speed and selection results. Explicit BF16/F32 selector construction and buffer guards fixed the seam; complete native shortlist/rescore tests now pass. Invalid receipts are excluded from gain claims. |
| Preserve the predictor's already-valid root KV | Kept | After drafting, rewind to the retained root and replay only accepted proposals against their preceding committed target hidden rows. Root-only acceptance avoids replay. This removes redundant state reconstruction; final numerical, generation-hash and prefix checks passed. No separate screened speedup is attributed to this change. |
| Cache-only predictor priming and catch-up | Kept | Preserve input fusion, normalization, projection, RoPE and KV writes; omit attention output, FFN and final normalization whose outputs are unused. Zero-draft rounds still prime predictor state. Exact restoration and lifecycle checks passed for each qualified head policy. |
| Consolidate target commit, predictor repair and last-hidden ownership | Kept | A three-draft round now needs five normal command completions instead of eight. The exact host sampler remains unchanged. The final combined build achieved the screened 51.483 tok/s result, but the separate contribution of wait consolidation was not isolated. |
| Fully GPU-side sampling/scheduling like the other engine | Investigated only | A private MLX diagnostic measured 55.32 tok/s over three 128-token sampled responses. It used GPU F32 sampling and omitted the full oMLX scheduler, custom QMM and coarse head. Its floating-point/RNG behavior, output length and implementation differ from the native six-response protocol, so it neither qualifies the target nor establishes seeded equality. A native replacement of the host F64 sampler was not implemented. |

## Draft depth and longer verification

| Approach | Outcome | Evidence and reason |
| --- | --- | --- |
| Fixed draft-depth screening | Three remains the recommendation | Depths one through seven were measured in a private longer-chain worker over two 128-token greedy responses for each of three prompts, in ascending/descending order. All 42 completion hashes matched. Two versus three was close and host drift remained visible; the experiment did not justify changing the production recommendation. Drafting still defaults to zero. |
| Seven drafts with target row reuse capped at four | Removed | Average advancement rose from 3.02 to 3.85 tokens/round on the pilot prompt, but target GPU time rose from 47–50 to 79–85 ms; throughput fell from 48–50 to 35–37 tok/s. Verification needed two target-weight scans, and longer acceptance did not offset the added cost. |
| Eight-row full-attention verification and explicit scalar-attention repair | Correct after repair; removed with the long path | The initial batched-attention candidate failed the numerical gate. Transposing prepared head-major queries and running the existing scalar core at each row's exact visible KV length restored exactness. Component comparisons and 49,167,360 whole-model logits then matched. Correctness alone did not justify retaining a slower long-depth feature. |
| Full eight-row weight reuse | Removed | Eliminating the second weight scan still did not make longer chains competitive. The unscreened depth sweep measured pooled medians of 38.633 tok/s at three drafts and 26.232 at seven. A small five-draft win on one prompt did not offset losses elsewhere. The experiment did not isolate register pressure, occupancy or compiler effects as the cause; no such causal claim is made. |
| Online adaptive draft-depth controller | Investigated only; not implemented | Request-local accepted-length histograms were added for analysis. No calibrated adaptive policy was implemented or qualified, and the fixed-depth results do not prove one cannot help. The final worker keeps explicit depths zero through three and does not route by observed timing. |

## Sources, related work and evidence limits

The inspected oMLX sources are pinned at
[`2165e75726ba83d9651dbaced61c97728da77bba`](https://github.com/jundot/omlx/tree/2165e75726ba83d9651dbaced61c97728da77bba),
including its batch generator and verification kernels. MLX 0.32.4 and the
separately pinned MLX-LM comparison revision are recorded in the initial report.
Source attribution and licenses are preserved in
[upstream provenance](../../provenance/UPSTREAM.md).

Earlier Stark Flash-Next/Lily work was consulted for kernel, sampler and state
reuse ideas. Its [optimization history](QWEN38_OPTIMIZATION_HISTORY.md),
[kernel study](QWEN38_KERNELS_2026_10_29.md) and
[exact-MTP study](QWEN38_MTP_COMBINATION_2026_10_31.md) concern different graphs
and checkpoints. Their rates and qualification results are not transferred to
this dense 27B artifact. No claim is made that every other inference engine was
benchmarked on this checkpoint.

Private receipts identify intermediate builds, synthetic kernel sweeps,
invalid candidates and rejected longer-chain sources. They retain source and
binary hashes; private model data and native validation artifacts remain
ignored. Public evidence contains aggregate results and hashes only:

- [Initial aggregate record](data/qwen38-27b-oq4e-mtp-2026-11-06.json).
- [Final aggregate record](data/qwen38-27b-packed-followup-2026-11-06.json).

Host interference was treated as a measurement problem. Earlier quiet-window
timeouts, CPU bursts and approximately 15% reference drift were not converted
into qualified performance claims. The final host-screened run had no excluded
attempts, and its six completion hashes matched the packing reference. It
qualifies the named final configuration; it does not isolate every retained
optimization or reproduce the public reference's unpinned pre-requantization
artifact and unavailable prompt corpus.
