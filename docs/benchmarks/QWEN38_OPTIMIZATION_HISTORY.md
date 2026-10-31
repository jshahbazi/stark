# Qwen3.8 optimization history

Compiled 2026-10-31 from the optimization conversation, dated validation records,
preserved private aggregate receipts and all reachable Git history through
`3d22b4dfb41aab4b10001bef6bbdf37a4f5cb86b`. This is an experiment ledger.
Original dated reports retain their source, workload, precision and validation
boundaries; this document does not retroactively strengthen them.

The current matched-checkpoint result is **66.8075 tok/s Stark versus 61.1711
tok/s MLX-Serve**, a **9.2142%** advantage by the requested maximum complete
request statistic. Both loaded identical stored weights and quantization,
without speculation. The earlier **79.52 tok/s** peak used Lily's different
quantization profile. The shared official checkpoint has not reached
77–80 tok/s, and cross-engine numerical or answer-quality equivalence remains
unestablished. See the [current result](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md).

## How to interpret the ledger

| Status | Meaning |
| --- | --- |
| Retained, model gain | Kept with a measured model-level improvement under the named controls. It does not imply every workload improves. |
| Retained, component gain | Exact or separately bounded controls and faster named component; a reliable complete-model gain is unproved. |
| Retained, correctness | Kept to implement the architecture, match a reference boundary or repair a defect. No isolated speed claim. |
| Inconclusive | Tested, but effect size, drift, order effects or controls prevent a dependable gain claim. |
| Rejected | Removed or never admitted after a numerical failure, regression or insufficient benefit. |
| Deferred | Source/research exists, with no completed qualification at the required scope. |
| Historical comparison | Useful measurement of the named executors; not an isolated optimization experiment. |

Unless specified otherwise, sampled studies use 4,096 input tokens, 256 outputs,
temperature 1, top-p .95 and seed 42. Top-k 20 and unbounded nucleus are separate
workloads. A normal useful decode rate counts the **255 advances after the
prefill draw**. The first cross-engine comparison instead used 256/output-time
for both engines; those figures are retained in their original convention.
The early F32 studies used short greedy, EOS-limited workloads and cannot be
placed on a single speedup curve with 4K sampled generation.

Same-loaded controls toggle a route between serialized, freshly reset requests
in one model process. They control a source of fresh-load variation, but do not
eliminate desktop scheduling, frequency or time dependence. Component timing,
instrumented GPU phase shares, standalone peaks, pooled rates and maximum
complete-request rates remain distinct evidence.

## Checkpoint and executor domains

| Domain | Fixed source and relevant difference |
| --- | --- |
| Pipe/F32 | `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, revision `b2c422f3c643e36f04227a64d61796b44a4b1029`; original standalone Rust/Metal F32 execution. |
| Lily fast path | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl`, revision `0f94842e3682b49acace4daeb4cf2e65a6f94f83`; 55 shards, 106,905,155,192 bytes. BF16 activations, F32 reductions/logits/GDN state and Q8 KV in the early fast-path studies. |
| Converted storage, same fast executor | Self-contained canonical Uzu text artifact derived from those Lily weights, weight-file SHA-256 `c493d4a3f876f6495f42127744b6415fe4cd3287f017fbc4b12ecd6d1245e657`; initially still executed by the specialized hybrid/Lily graph. |
| Full standard Uzu executor | Same converted codes and affine coefficients, with standard `Engine`/`LanguageModel`, `WeightMatrix`, backend commands, scratch and GPU sampler; different documented arithmetic and RNG. Experimental work preserved at `174854991e022ff7283dfa6f0f951114de708ac1`. |
| Shared official checkpoint | `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`, revision `788c07a8725642bccf13a0a3b8a21954ab84b4f2`; both engines load one directory, 113 files, 107,326,639,344 bytes. |
| iQ-MLX 4.7 investigation | Revision `a0e89db090a3e4b2e8ea2118fe9d0324d3854aca`; metadata investigation only. Published M5 Ultra/MTP results were outside this machine and no-speculation comparison. |

Lily has Q4 GDN and Q8 embeddings; the shared official pack has Q8 GDN and Q4
embeddings. HC injection/shared-expert gates also differ (affine Q8 versus dense
BF16). Routed experts remain Q4/group64 and n-gram rows Q4/group32. Matching
total bytes does not establish equivalent quantization or model quality.

## Evidence map

Short labels in the tables refer to these unchanged reports. Each report
contains more exact methods, private receipt hashes and executable identities.

| Label | Source record |
| --- | --- |
| F1 | [Native projection/memory optimization](GLM_QWEN_NATIVE_OPTIMIZATION_2026_10_28.md) |
| F2 | [GPU stage chaining and packed matvec](GLM_QWEN_NATIVE_GRAPH_2026_10_28.md) |
| F3 | [Complete decode graph](GLM_QWEN_DECODE_GRAPH_2026_10_28.md) |
| MP | [Mixed precision and Lily integration](QWEN38_MIXED_PRECISION_2026_10_28.md) |
| S | [Sampled BF16 generation](QWEN38_SAMPLED_BF16_2026_10_28.md) |
| K | [MoE, HC and GDN kernels](QWEN38_KERNELS_2026_10_29.md) |
| ST | [Stability and exact sampling](QWEN38_STABILITY_2026_10_29.md) |
| PW | [Automatic/High Power controls](QWEN38_POWER_MODES_2026_10_29.md) |
| FU | [Residual fusion and parallel sampler](QWEN38_FUSION_2026_10_29.md) |
| IQ | [iQ-MLX metadata investigation](QWEN38_IQ47_INVESTIGATION_2026_10_29.md) |
| H | [Metal hotpath trials](QWEN38_METAL_HOTPATHS_2026_10_29.md) |
| U1 | [Uzu storage conversion](QWEN38_UZU_FORMAT_2026_10_29.md) |
| U2 | [Converted loading and exact nucleus](QWEN38_UZU_SPEED_2026_10_30.md) |
| U3 | [Preserved full-Uzu implementation](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/QWEN38_UZU_EXECUTION.md) |
| U4 | [Preserved full-Uzu tuning failures](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/benchmarks/QWEN38_UZU_REJECTED_TUNING_2026_10_30.md) |
| U5 | [Preserved pause/cleanup decision](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/archive/runs/2026-10-30-qwen38-uzu-paused-checkpoint.md) |
| C1 | [Different-pack Stark/MLX comparison](QWEN38_STARK_MLX_SERVE_2026_10_30.md) |
| C2 | [Direct external adapter controls](../archive/runs/QWEN38_EXTERNAL_MLX_ADAPTER_CONTROLS_2026_10_30.md) |
| C3 | [First same-checkpoint comparison](QWEN38_SAME_CHECKPOINT_STARK_MLX_SERVE_2026_10_30.md) |
| C4 | [Final matched optimization and comparison](QWEN38_MATCHED_OPTIMIZATION_2026_10_30.md) and [aggregate JSON](data/qwen38-matched-optimization-2026-10-30.json) |

## 1. Original F32 execution, 2026-10-28

The user's approximately 22.94 tok/s starting point was already the result of
several earlier native-engine changes. F1–F3 record these on the Pipe/F32
checkpoint with 54 input tokens, a 512-token context and a 31-token EOS-limited
greedy continuation. Warm medians exclude the first request. Output controls
match within each stated reference/candidate comparison.

| ID | Attempt | Outcome and decision | Evidence/source |
| --- | --- | --- | --- |
| F01 | Native baseline and first optimization bundle | **Retained, model gain:** 0.672→4.318 tok/s decode; prefill .757→4.451. The profile initially found 917 submissions and 6.02 GB compressed uploads per Qwen token. The bundle removed those repeated host costs; it is not an isolated measurement of each subchange. | F1; `8056d605` |
| F02 | Replace power-of-two packed-index divisions with shifts/masks | Same addresses and reduction order; isolated four-step Qwen GPU control 1.562→.749 s. Retained inside F01. | F1 |
| F03 | Owned mapped Metal views and bounded selected-expert ranges | Avoid repeated upload/copy of immutable weights while keeping mapped owners alive through completion. Retained; no independent complete-model rate. | F1 |
| F04 | Batch independent projections and omit discarded prefill readouts | Reduced waits and work whose vocabulary readouts were unused. Retained inside the bundle, with matching final readout/output controls. | F1 |
| F05 | Model-local immutable Metal registration FIFO, 8 GiB cap | **Retained, model gain in adjacent short controls:** 3.599→4.708 tok/s. This is distinct from the rejected selected-expert FIFO below. | F1 |
| F06 | Specialize packed kernels by storage, bit width, group and parameter dtype | Removed repeated runtime decoding/admission work while preserving F32 reference controls. Retained inside F01/F2. | F1–F2 |
| F07 | Four-output-row scheduling | **Rejected:** synthetic large IQ2/affine wins did not transfer; actual Qwen Q4 expert shapes regressed about 5%, floating/Q8/affine8 also lost. Shared GLM control 2.589→2.312 tok/s. | F1 |
| F08 | Bounded 200 µs completion polling | **Rejected:** Qwen blocking 3.599 versus polling 3.495 tok/s, CPU time 1.01→2.92 s. Blocking waits retained. | F1 |
| F09 | Chain FFN, HC, GDN state/history and intermediate buffers on GPU | **Retained, model gain:** 4.3180→12.5153 tok/s, prefill 4.4515→12.6666. Routing and remaining scalar operations still used the CPU at this stage. Previous baseline was reused rather than interleaved. | F2; `018f89c5` |
| F10 | Register complete expert banks under strict memory admission | Retained only if rank-3 banks fit 75% of recommended working set after the static reservation, with bank-size/offset guards. Unsupported cases keep transient ranges. No separate Qwen throughput attribution. | F2 |
| F11 | Eight-values-per-lane packed dot and shape-specific concurrent FFN | Retained within source-reference numerical gates; reduction association changes. Synthetic Q4 gate matrix 25.26→8.48→1.85 µs for scalar/vector/concurrent scheduling; Q8 down concurrency regressed 4.97→5.94 µs, so concurrency is not universal. | F2 |
| F12 | Generic eight-adjacent scalar decode / wider shared GLM blocking | Generic scalar variant rejected: Q5/Q6 9–16% slower and readout about 10% slower. Block-aware Q5/Q6 paths were useful for GLM; a Q8 128×4096 case regressed over 4× and retained scalar fallback. These are shared-backend findings, not Qwen Q4 gains. | F2 |
| F13 | Looser complete-bank memory admission | **Rejected shared GLM trial:** no first request after 310.75 s; no throughput result. Did not justify relaxing Qwen/GLM admission. | F2 |
| F14 | Sharded command-attached residency hints | **Rejected:** exact Qwen output controls, but 12.3024→7.7890 tok/s. No-copy registration/residency is not a speed guarantee. | F2 |
| F15 | Selected-slice FIFO cache | **Rejected shared GLM trial:** 7.78835→7.02956 tok/s. Kept separate from successful immutable registration caching. | F2 |
| F16 | Encode complete hybrid decode graph with one completion per token | **Retained, model gain:** Qwen 11.214→22.936 tok/s, prefill 11.603→23.981; checked views stay alive through deferred execution. This produced the reported 22.94 starting point. | F3; `28004157` |
| F17 | SIMD RMS reduction with source-ordered F32 fallback | Retained parallel route with existing source gates; `STARK_HYBRID_ORDERED_RMS` preserves ordered reference behavior. Sum-tree changes are explicit numerical policy. | F3 |
| F18 | First graph scheduling candidate | **Rejected shared GLM control:** 7.656→4.261 tok/s after losing concurrency/ordering advantages; corrected before the retained graph. Not a Qwen throughput estimate. | F3 |

## 2. F32 tuning, mixed precision and the Lily fast path

The user explicitly allowed mixed precision and requested a policy update.
Full F32 execution remained reference evidence rather than a production
requirement. The Lily integration preserved the upstream revision and licenses;
it did not call an external Lily process or server.

| ID | Attempt | Outcome and decision | Evidence/source |
| --- | --- | --- | --- |
| P01 | Further same-Pipe F32 graph/kernel bundle | **Retained, model gain:** 23.1563→43.7012 tok/s (+88.72%) on the short greedy workload, with matching artifact and completion hash. Includes independent projection/norm scheduling, cached pooled index keys, split long QSA, blocked GDN state, fused finite checks, wide packed loads and cached geometry admission. | MP; `8dc6ccd0` / `0bb1df44` |
| P02 | Lossless header-aligned Safetensors copy | 45.6490 tok/s observation; all 11 tensor-payload hashes preserved and only JSON padding changed. Useful alignment evidence, not a new quantizer or a randomized isolated speed estimate. Temporary duplicate later removed. | MP |
| P03 | Two-stage F32 HC fusion | **Rejected:** all four targeted tests passed, hidden values exact and injection within unchanged bound; warm completed-command component time 43.130→50.729 µs and 40.781→50.643 µs (17.6–24.2% slower). Cold compilation advantage was excluded. | [Archived experiment](../archive/experiments/qwen38-f32-hc-2026-10-28/README.md) |
| P04 | BF16 execution policy | **Retained, correctness/policy:** BF16 activations, F32 reductions/logits/recurrent state, Q8 attention cache; unchanged stored codes within a checkpoint. New precision routes require matching validation. F32 records remain preserved. | MP; `AGENTS.md` |
| P05 | In-process Lily architecture/core integration | **Retained:** pinned `e22d58fa7cc488493fb1b5d810a198bd29730aeb`; original 14 shaders initially unchanged, resident experts and paged n-grams. 64/64 complete F32 readouts matched untouched Lily over short/4K teacher controls; reset/recovery passed. The switch from Pipe to Lily changed checkpoint packing, so 43.7→77 is not a same-weight engine speedup. | MP |
| P06 | Safe CPU encode-ahead for known token execution | **Retained, model gain on Lily:** synchronous greedy 74.0307→77.1836 tok/s at 4K/256. Validates/stages only the actual token, uses exact remaining budget and drains failure/cancellation work. It never proposes speculative tokens. | MP |
| P07 | Initial BF16 phase profile | Diagnostic only: MoE 31.4%, HC 22.6%, GDN 21.4%, QSA 15.6%, head 8.1%; established the requested MoE→HC→GDN work order. Profiled timing is not production throughput. | S |

## 3. Exact CPU sampling and validation work

CPU selection still executes when the model forward uses Metal. These changes
therefore apply to the fast Metal executor's host sampler. Exact ordering,
probability arithmetic, CDF traversal and RNG boundaries matter independently
of the GPU forward kernels.

| ID | Attempt | Outcome and decision | Evidence/source |
| --- | --- | --- | --- |
| S01 | Partial top-k selection instead of complete sort | **Retained, model gain in repeated window:** top-k20 warm median 50.6156→75.0780 tok/s (+48.33%). Initial 47.0501→68.3783 observation is a separate window. Full 256-output hashes match within mode. | S; `d13e5cde` / `80474e05` |
| S02 | Stable four-pass 8-bit radix full ordering | **Retained:** real-readout full sort 3.874821→1.146534 ms (70.41% less); every one of 256 complete orders exact. Descending F32 total order, ID ties and signed zero preserved. Repeated-window nucleus median 49.7349→65.5722 with the combined sampler/scheduling changes. Earlier 36.0387 baseline is not interchangeable. | S |
| S03 | Sampled encode-ahead | **Retained:** pre-encode command structure, then stage the CPU-selected token; no unused argmax, no speculative work, no RNG draw on drained cleanup. Part of S01/S02 end-to-end bundle. | S |
| S04 | Small top-k≤64 contiguous prefix scan | **Retained, component gain:** top20 CPU .274468→.072637 ms (73.5% less), with bounded replacement work and exact partition fallback. | ST; `9204b0b9` / `174d35ef` |
| S05 | Three 11-bit radix passes and packed key/ID | **Retained, component gain:** 1.118435→.635396 ms (43.2% less), all 256 full orders exact. | ST |
| S06 | Parallel F64 exponential mapping for large vocabularies | **Retained, component gain:** ≥65,536 candidates, lazy worker-owned pool up to four CPU threads; sum/filter/CDF/RNG stay serial. 63,569,920 probability bits and 768 filter states exact. Mapping .920267→.379962 ms; total sampler 2.428301→1.869726 ms. Same-loaded nucleus pooled 57.724→58.454 (+1.3%), median +1.7%; mixed directions and drift leave the full-model gain inconclusive. Top-k20 correctly stays serial. | FU; `d6fbd8b7` |
| S07 | Four-bank CPU radix histograms | **Rejected:** exact standalone controls, but random-input sort 544.888→617.852 µs (+13.4% time), hot-low-digit 907.078→1,101.575 µs (+21.4%). | H |
| S08 | SIMD finite-value validation | **Retained, component gain:** exact predicate, 32-value chunks permit NEON; top-k guard mean 38.73→8.36 ms/request, nucleus 53.71→9.70 ms (elapsed timers include descheduling). Same-loaded complete generation pooled −.84% top-k / +1.42% nucleus, mixed medians/blocks; no reliable total gain. | H; `28510ca7`, reported in `edf92a02` |
| S09 | Typed `as_chunks::<32>()` final guard | **Retained, separately validated:** 16,128 differential comparisons; same hot NEON scan as the tested `chunks_exact` version. Synthetic vocabulary scan 63.125→14.500 µs. Final smoke medians 76.27/71.34 are separate-load validation, not paired gains. | H |
| S10 | Simple sampler-buffer reuse | **Rejected:** same-loaded pooled −.55% top-k and −1.40% nucleus. | U2; `90020127` |
| S11 | Keep packed IDs as an independent sampler change | **Inconclusive/not independently retained:** about 3.15% nucleus CPU reduction, noisy generation controls. | U2 |
| S12 | Exact Top64 nucleus proof shortcut, V4 | Proves the reference stopping decision with an outward-rounded interval; otherwise full sort. Frozen-readout 888.83→350.71 µs (32 tokens), 910.94→350.29 µs (4K). Full generation used shortcut 194/256 draws. Same-loaded nucleus 55.56→57.51 (+3.51%), but cycle deltas −3.56% to +9.77% and falling clocks make the model gain inconclusive. | U2 |
| S13 | Final V5b proof shortcut with rounded-probability reuse fallback | **Retained, component gain:** 899.04→350.02 and 920.48→349.77 µs; probability bits, prefix, IDs/RNG and scratch cleanup exact. Final same-loaded nucleus 55.648→55.634 tok/s, effectively unchanged; unchanged top-k control 57.743→57.249. Lily-scoped only; bounded scratch clears on draws/errors/request boundaries. | U2; `90020127` |

The early parallel-sampler production binary had a later-discovered build
provenance limitation, recorded below. Its exact mapping ABBA worker and CPU
controls were separately audited. The 77.55/72.40 standalone follow-up did not
prove that parallel exponentials alone produced that throughput.

## 4. MoE, HC and GDN kernel trials on the Lily checkpoint

Rows K01–K10 precede later hotpath research. The accepted graph preserved the
Lily seeded continuations. Its same-loaded comparison ultimately measured only
**+1.93% top-k** (61.1398→62.3204 pooled) and **+.05% nucleus**
(64.0755→64.1106). It did not establish a reliable ≥3% full-model gain.

| ID | Attempt | Outcome and decision | Evidence/source |
| --- | --- | --- | --- |
| K01 | Masked-nibble MoE m1 unpack | **Rejected before production:** 24,576 mismatches among 1,566,720 finite BF16 word cases. | K; `908ae21f` / `eba4cafe` |
| K02 | MoE row packing, 2/4/8 SIMD groups | **Rejected:** exact 81,972 outputs but all selected component chains were flat/slower: 87.792→88.854, 203.229→203.778 and 203.229→206.958 µs in their own windows. | K |
| K03 | Existing m1 HC down/up pair | **Rejected:** 28 synthetic cases passed, but full-model teacher and both sampled hashes changed. Synthetic parity was insufficient. | K |
| K04 | Original HC down + separate activation + cached up | **Rejected:** restored model hashes but three dispatches gave 59.256094→59.256389 µs/read. | K |
| K05 | m1 down + cached original HC up, two dispatches | **Retained, component gain:** original dot/rounding boundaries; 57.882195→54.757597 µs/read (−5.40%), 28 cases/67,416 bytes exact. Full-model impact limited by the combined same-loaded result above. | K |
| K06 | GDN retain BF16 output in thread-local registers | **Retained, component gain:** removes barrier/256-byte threadgroup storage; 17.872397→17.666233 µs/layer (−1.15%); 12×16-step controls over 341,065,728 state/output bytes exact. | K |
| K07 | Q4 GDN projection + four-tap convolution | **Retained, component gain:** guarded two-row geometry, one history writer; 58.126809→54.857205 µs/layer (−5.62%), six 16-step cases/5,612,352 bytes exact. | K |
| K08 | Cached MoE input sums | **Rejected:** 42 exact cases/258,034 bytes; chain 56.740→58.094 µs, intrinsic prepared-input control essentially unchanged. | ST |
| K09 | Guarded masked unpack with fallback | **Rejected:** first helper-passing variant had 48 BF16 kernel mismatches; original fallback restored parity. Final 1,828,864 word cases and 42 kernel cases passed, but chain 56.665→61.406 µs. | ST |
| K10 | Cached scaled F32 MoE inputs | **Rejected:** exact controls, extra 10,880-byte scratch, chain 90.308→90.856 µs; no benefit. | ST |
| K11 | MoE down/combine → HC residual-write epilogue | **Retained, component gain:** removes 48 dispatches, preserves BF16 branch round and F32 slot order/shared gate; 36×16-step cases/5,241,600 bytes exact. Streaming component 31.599068→29.845269 µs/layer (−5.55%, about .084 ms/token across 48 layers). Same-loaded top-k −2.22%, nucleus +1.18%, mixed blocks; no dependable total gain. | FU; `d6fbd8b7` |
| K12 | GDN/QSA output projection → HC injection | **Rejected:** 54×16-step cases/5,062,080 bytes exact, but only about .91% component reduction (.0123 ms/token across 48 layers); complexity and dispatch removal did not justify retention. Separate prototype 78.0272/68.6412 versus production 77.5500/72.4027 was not a paired gain. | FU |
| K13 | HC gain/RMS prepass cache | **Rejected:** three dispatches, component 33.627→40.932 µs/read. Same-loaded pooled 74.21→72.68 top-k, 69.73→68.11 nucleus. | H |
| K14 | Literal HC-down dimensions | **Inconclusive/not retained:** 288 exact cases/1,347,552 bytes; component down+up 25.039→22.185 µs. Same-loaded top-k +2.2%, nucleus +.3%, mixed cycles under substantial observed external activity. | H |
| K15 | Virtual-lane MoE reduction | **Rejected:** initial mapping failed 5,570/16,384 rows; hardcoded shuffle still failed 1,273. | H |
| K16 | Dynamic-pair MoE reduction | Passed device-bound controls but depends on observed compiler `simd_sum` order, with no portable matching-order guarantee. **Rejected:** same-loaded top-k 65.13→63.07 pooled, nucleus 61.69→61.93; mixed blocks/no reliable gain. | H |
| K17 | GDN register reuse v1/v2 | v1 **rejected** on first exact state control. Corrected v2 exact over 341,065,728 bytes; median 16.862271→16.665507 µs/layer, only .0071 ms across 36 layers. **Component-only**, insufficient model evidence for retention. | H |
| K18 | Packed-Q4 N64 gate/up | **Rejected:** layout/numerical controls passed; full component 65.125→238.329 µs/layer (3.659× slower). | H |
| K19 | Packed-Q4 split-K8 gate/up | **Rejected:** controls passed, full split/reduce/activation/down/HC chain 67.472→145.360 µs/layer (2.154× slower). | H |
| K20 | Packed-Q4 N16/N32 | **Deferred:** sealed source/patches, not built or tested. No speed or quality result. | H |
| K21 | Grouped NAX MoE | **Rejected:** finite synthetic outputs but about 50% slower; BF16 agreement only about 55–70%, branch/HC relative L2 roughly .0075. No model-quality acceptance attempted. | H |
| K22 | GS64 specialization | **Inconclusive:** separate short windows gave 74.598 top-k /66.075 nucleus medians; no same-loaded causal comparison, not qualified for retention. | H |
| K23 | Stride-aware standard-Uzu GEMV port | **Rejected:** several geometries/Metal versions failed to improve actual Q4 GDN, lost on QSA QKV; small synthetic gains did not transfer. Fused compact-expert gate/up port also lost. | U2 |
| K24 | BF16 attention-cache experiment | **Rejected:** materially changed full-vocabulary readouts. Shipped Q8 cache stayed in place. | U2 |

Separate-worker ABBA during K05–K07 showed large reversals (top-k candidate
64.1763 versus baseline 72.9552 pooled, nucleus 61.4601 versus 48.0785).
The same-loaded result is the stronger attribution control; those historical
windows remain preserved rather than selecting only the favorable peak.

## 5. Throughput instability, host policy and benchmark controls

The user initially defined consistency to include normal desktop multitasking,
then explicitly closed apps and later reserved the machine for this work.
The closed-app condition is a user-controlled condition, not independent proof
of zero other GPU work. No benchmark discarded slow measured requests.

| ID | Attempt | Outcome and decision | Evidence/source |
| --- | --- | --- | --- |
| H01 | Repeat unchanged worker before optimization | Top-k median 56.6968, range 53.3238–61.0991 without source/weight changes: instability reproduced independently of a candidate. | ST |
| H02 | CPU/VM/GPU-client and rootless clock telemetry | Found throughput/frequency/time correlations and WindowServer/other-client overlap. Raw GPU-client units do not establish utilization percentages or exact causal cost. No pageouts/swapouts in the key sustained sampler control. Retained diagnostic capability. | ST |
| H03 | Per-request process activity assertion | Same-loaded top-k 53.5065→52.4748, nucleus 49.4293→48.3627; **no speed benefit**. Retained for request-lifetime semantics, without a throughput claim. | ST |
| H04 | Thread QoS override | **Rejected:** worker already `USER_INTERACTIVE` (0x21); requested `USER_INITIATED` would lower it. Separate timings did not establish a gain. | ST |
| H05 | Explicit serial feedback queue | **Not retained:** default→queue 70.0958→70.2200 top-k, 66.5940→67.2694 nucleus; callbacks had lower QoS and about 1.6 µs latency difference, too small/noisy. | ST |
| H06 | Sustained alternating sampled control | Nine measured requests/case: top-k 72.77 pooled, range 70.34–75.44; nucleus 67.63, range 65.14–70.82. Initial warmup 77.89 at 1,614 MHz preceded roughly 1,220 MHz later clocks. Correlation does not prove thermal/power causation. Target not sustained. | ST |
| H07 | Temporary AC High Power, then restore Automatic | **No demonstrated benefit:** Automatic-before 72.8832/67.3876 pooled top-k/nucleus; selected High Power 66.7898/62.5704; restored Automatic 65.4143/61.4384. All 60 hashes exact, no measured request ≥77. Different desktop activity, load order, firmware flags and possible late lock prevent attributing slowdown to High Power. Original preferences restored and verified. | PW; `6f953e8c` / `565a9a5f` |
| H08 | Quiet-window repeated baseline and stage profile | Nine measured requests/case: 73.62 pooled top-k, 69.85 nucleus. Instrumented GPU shares about 30% MoE/23% HC/22% GDN; clocks near 1,305/1,324 MHz. Diagnostic spans are not production throughput or additive to blocking wait. Target still unmet. | H |
| H09 | Per-checkout Cargo target isolation | **Retained, correctness/provenance:** shared target reused an experimental core after root source restoration. Earlier sampler worker contained a default-on private MoE flag without a setter; computation was production-equivalent but clean provenance was not established. Canonical rebuild removed hooks, audited 900 inputs and restored worker `c909c5d7…`. Its later 64.0185/60.0075 smoke window was not a performance regression attribution. Future experiments use separate targets. | FU; `f669f721` |

## 6. Storage conversion and full standard Uzu execution

The first conversion solved the artifact contract and retained the existing
compute graph. The later user request required the standard Uzu executor.
These are two separate attempts and two separate validation domains.

| ID | Attempt | Outcome and decision | Evidence/source |
| --- | --- | --- | --- |
| U01 | Canonical self-contained text conversion | **Retained storage support:** 3,431 text planes verified: 1,046 exact codes, 1,834 inverse-transposed affine planes, 551 direct planes; zero padding checked. Vision/MTP planes omitted because unused. Full-vocabulary F32 readouts at 32/4K exact, source-directory-denied execution passed. No requantization. | U1; `66b29b16` |
| U02 | 8 MiB converter workspace | **Interrupted/replaced:** large parameter planes triggered row-at-a-time fallback with excessive tiny reads. 64 MiB workspace admitted contiguous reads and completed the same layout. | U1 |
| U03 | Native-source versus converted-storage ABBA | **No established speed gain:** medians 63.53/57.43 top-k/nucleus native versus 53.61/54.61 Uzu; block rates/frequency fell strongly over time in both arms. All 40 outputs exact; equal observed Metal allocation. Additional monitored series likewise did not isolate a storage effect. | U1 |
| U04 | Ordinary standard-matrix component comparison | Q4 GDN GPU 28.028→24.859 µs, Q8 attention 61.430→56.958 µs; host-inclusive times slightly worse. Outputs not bit-identical (different reductions), bounded oracle only 64 rows. **Component research**, not integrated complete-graph gain. | U1 |
| U05 | NEON 8×8/64-row integer transpose while loading affine planes | **Retained, startup gain:** exact carrier bits including exceptional encodings/tails/error order. Fresh-process ABBA process-to-ready median 20.535→18.091 s (−11.90%). Decode repeats no transpose; warm decode controls did not show improvement. | U2; `90020127` |
| U06 | Full standard `Engine`/`LanguageModel` Qwen decoder | **Implemented, experimental:** standard canonical `WeightMatrix`/matmul, backend buffers/scratch/commands, complete 48-layer HC/GDN/QSA/MoE/PLE graph, request-owned state and GPU sampler. Production Lily dependency removed on that branch. Initial short sampled decode 14.1657 tok/s; not accepted as fast-path replacement. | U3; `a6c8b1bd`; private aggregate A01 below |
| U07 | Parallel top-10/512-expert router | **Retained on experimental branch, large diagnostic gain:** replaced serial single-lane scan while preserving tie/score rules; short 32/256 warm median 14.1657→51.5071 tok/s; 4K 44.2643. Separate three-request windows, one warmup; not randomized isolated long-run attribution. | U3/U5; A01–A03 |
| U08 | Batched/layerwise prefill, bounded expert gathers and requested readouts only | **Retained experimental architecture:** chunks ≤16, gathered rows in groups ≤8, causal GDN/cache updates; no CPU projection/routing/recurrence fallback. Final 4K prefill remained only about 135–185 tok/s, far below fast executor. | U3/U4 |
| U09 | F32 split-K partial correction | **Retained correctness on experimental branch:** original BF16 partials lost cancellation before reduction (136/1,088 outputs outside derived bounds); F32 partials and dtype cache key repair the defect. Separate cancellation/branch controls passed; post-fix 4K 43.5337 tok/s is not a speed gain. Historical broad logs lacked immutable source/binary receipts, explicitly limiting final qualification. | [Failure](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/archive/runs/2026-10-30-qwen38-split-k-precision.md), [correction](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/archive/runs/2026-10-30-qwen38-split-k-correction-validation.md); A04 |
| U10 | Lexical-cutoff parallel router refinement | **Retained experimental:** 4K warm 43.9677 decode/136.235 prefill; not a dependable speed delta from preceding separate 43.5337 window. | U4; A05 |
| U11 | Q4 gathered projection tiles | **Rejected:** exact component pairs, nominal +5.15% gate/up/+11.03% down with large order effects; 4K model 43.9677→44.1032 (+.31%), prefill −3.12%. Selector removed. | U4 |
| U12 | Q8 HC down/up/injection tiles | **Rejected:** exact controls, rotating component +25.56%/+17.93%/+38.75%; full model only 44.3018 tok/s (+.76% separate window). Extra selector/geometries removed. | U4 |
| U13 | Execution-only WAR barrier without cache visibility flush | **Rejected:** preserved pending writes/device barriers where needed; 43.6455 tok/s (−.73% separate window). SDK motivation did not establish a speed benefit. | U4 |
| U14 | Selective direct-integer/rowwise GEMV, all admitted components | **Rejected and reversed:** synthetic exact gates did not transfer. Same-loaded 32/256 false/true outputs repeat within arm but differ across arms; 48.0919→48.1829 tok/s (+.19%, approximate interval −7.21% to +8.07%) is not accepted performance evidence. Separate 4K 44.6945→48.0761 windows also have different digests. | U4 |
| U15 | Independent affine scale/bias diagnostics | **Correctness investigation:** 30 synthetic cases retained derived bounds, no nonfinites/cache-repeat failures, but 40,358 exact mismatches exposed the inadequacy of earlier centered-bias fixtures. HC-up excluded from subsequent exact-route trial. No tolerance was relaxed. | A07 |
| U16 | Mask5 selective GEMV: HC-down rowwise + MoE-down direct/rowwise, HC-up off | **Inconclusive:** same-loaded 32/256, 8 ABBA quads/16 measured requests per arm, cross-arm/repeat hashes exact; 51.3931→52.2774 pooled (+1.72%), approximate paired interval −1.87% to +5.21%. Selector counts describe CPU route assignment, not GPU completions. No production retention or 4K gain established. | A06 |
| U17 | Wider K lanes | **Rejected/not adopted:** independent synthetic bounds passed; most Q8 and Q4/32 cases regressed. Width changes alter reduction association. | U4 |
| U18 | Standard GEMV fused gate/up | **Rejected/not adopted:** synthetic projection/activation controls passed; normal routed Q4 regressed, alternate geometry did not isolate fusion, shared Q4/Q8 did not establish gain. No model gain measured. | U4 |
| U19 | Four local dot accumulators, scalar input sum (ILP style1) | **Component-only/not adopted:** actual-Q4 corrected harness, QKV +3.20%, Q4 down −6.01%; HC Q8 about −2.64% to −4.17%. Strong first/second-arm effects. No complete-model qualification. | A08 |
| U20 | Four local dot and input-sum accumulators (style2) | **Component-only/not adopted:** Q4 QKV nominal +26.26%, z +19.05%, a/b about +14.9%, gate/up +13.09%; MoE down −4.50%, Q8 HC about −2.8%. Association deliberately changes; synthetic output agreement is not a model parity guarantee. | A08 |
| U21 | Four group-result banks (style3) | **Rejected component direction:** QKV +6.53%, z −16.19%, a/b −27.8/−28.5%, HC-up −27.11%, MoE-down −25.22%; changed QSA/head values under valid independent bounds. | A09 |
| U22 | Wider lanes + rowwise scheduling | **Component-only:** original-width Q4 rowwise nominal +.86% to +30.72% over six shapes, zero changed fixture outputs; wider Q4/Q8 arms had substantial regressions and some Q8 output changes. No sampled-generation qualification or retention. | A10 |
| U23 | Wider lanes + local4 + rowwise (style4) | **Component-only:** original-width Q4 mixed −2.34% to +26.39%; Q4 width32 mostly loses, Q8 HC-up width32 about −49.2%. Same-width style2/style4 equality controls isolate scheduling only within those synthetic fixtures. | A11 |
| U24 | Exact GroupOutput→OutputGroup affine coefficient layout | **Not adopted:** coefficient/code bits and complete F32/BF16 controls exact; expert pooled nominal +2.94% gate/up/+1.34% down, smaller geometric effects and much larger order strata/self-control noise. QKV −1.12%, HC-up −2.26%. Canonical format/production loading unchanged. | A12 |
| U25 | GDN float2 state loads/two values per thread | **Rejected exact variant:** original full-kernel state parity gate failed despite written per-component order; compiler reassociation differed. Separate scalar-pair/analytic/fixed-F64 investigations retained the failure. Exploratory full-kernel run later about 5.2% slower with changed F32 state; no model accuracy threshold or adoption. | A13–A15 |
| U26 | New Q4 rowwise and local4 production proposals | **Deferred:** separately prepared narrow policies and independent F64/bounds/poisoning/cache tests; no completed sampled-generation experiment before pause. They do not inherit U14/U16 results merely from related source. | U5; private proposal receipts |
| U27 | Full-vocabulary and incremental Uzu numerical controls | **Incomplete qualification:** experimental selective-GEMV worker: 35/35 finite, 34/35 top1, max KL .266118, max TV .287455; no exact F32 readouts. Later retained-path 128-M1 controls: 15/16 top1, max KL .0836745, max TV .198468. These are different workers and sampled checkpoints, not quality passes. | U5; [experimental record](https://github.com/jshahbazi/stark/blob/174854991e022ff7283dfa6f0f951114de708ac1/docs/archive/runs/2026-10-30-qwen38-experimental-full-vocabulary.md) |
| U28 | Qwen-only 65,536 context admission | **Implemented/deferred runtime:** focused source/worker/client tests pass; actual 65,280-input+256-output model benchmark/allocation remains unrun. Other models keep their limits. | U3/U5 |
| U29 | Preserve full-Uzu work and return to fast executor | **Decision:** same converted artifact, separate warm windows 44.6945 Uzu versus 77.8234 existing executor (about 42.6% lower); different numerics/RNG, no cross-executor hash claim. User paused migration, branch preserved at `17485499`. | U5 |
| U30 | Remove redundant models/build experiments after preservation | User-authorized cleanup; original Lily control 77.1778 tok/s with exact prior sample hash proved converted copy unnecessary for active route. Canonical Uzu artifact, old Pipe duplicate and rebuildable targets removed; compact metadata/source/patch/aggregate identities retained. No inference optimization is attributed to deletion. | U5; `17485499` |

The quantization correction in the private Uzu component inventory matters:
early “GDN Q8” component cases were synthetic Q8 simulations, while the pinned
Lily artifact's GDN projections are Q4. Actual-Q4 ILP and wide/rowwise runs were
recorded separately. Simulated Q8 results are not evidence for speeding up
Lily's GDN. No claimed register counts are hardware compiler/spill measurements.

## 7. External-engine research and equivalent-pack comparisons

| ID | Attempt | Outcome and decision | Evidence/source |
| --- | --- | --- | --- |
| C01 | Compare upstream Lily and other Apple-Silicon techniques | Pinned source/reference comparison motivated BF16, GPU graph/fusion, sparse QSA and host sampling work. Published throughput is not a local replicated result. The [literature review](../archive/runs/2026-10-28-qwen38-performance-literature-review.md) preserves source/date boundaries. | MP/S/H |
| C02 | iQ-MLX 4.7 pack / advertised higher rate | **Metadata-only/deferred:** card's 157 tok/s median (149–162) used M5 Ultra and MTP. Different device/speculation precludes the requested comparison. Discovered external table, folded norms, BF16 gates and per-tensor widths; no full pack load then due storage. Fresh existing-Lily control 72.08 greedy/64.92 sampled median did not qualify iQ-MLX. | IQ; `51bcada8` |
| C03 | Load Lily directly in stock MLX-Serve | **Failed compatibility:** `EmbeddedPleIndex` branch treated Lily n-gram spelling as NVFP4 and rejected affine planes. No stock-MLX Lily throughput established. Supported official mixed pack chosen under user's equivalent-quantization permission. | C1 |
| C04 | First serialized Stark/Lily versus MLX/official ABBA | **Historical different-pack comparison:** top-k20 max 79.52 versus 61.00, pooled 74.13 versus 58.53; 10 measured requests/engine, other apps closed. Different GDN/embed/gate widths and distinct timers/RNG preclude pure engine-efficiency/quality claim. User later requested replacement on website with matched weights. | C1; `19b31439` |
| C05 | Website's six-workload Qwen3.5 benchmark adapted to Qwen3.8 | **Deferred:** the published Qwen3.8 comparison remained a controlled 4K/256 sampled decode experiment, not the website's complete six-workload protocol. No six-workload Qwen3.8 result is inferred. | Conversation/C1–C4 |
| C06 | Direct official affine/external-table checkpoint adapter | **Retained, correctness:** aliases preserve original byte spans and per-tensor Q4/Q8/BF16; folded-gain variants, dense small gates, exact PLE hash controls and mapped external table. Found missing fused-attention folded-gain case, fixed before final component build. No conversion/requantization. Original Lily fallbacks remain. | C2; `434beaf6` |
| C07 | Complete official checkpoint download/integrity | Transfer paused/resumed as requested; eventually all 113 files/103 LFS payload hashes verified before timings. Downloads/checksums/builds finished and engines serialized. This is a prerequisite, not a throughput change. | C3 |
| C08 | MLX whole-table warming versus Stark demand prefetch | MLX warmed its 32,000,153,976-byte n-gram table before requests, `/props` completion verified. Stark adapter uses demand-driven row prefetch; available whole-table preload was not wired in. No controlled whole-table-warm Stark trial or speed gain established. Reclaimable page cache is not resident GPU allocation. | C3 |
| C09 | First exact-checkpoint ABBA | **Historical matched baseline:** Stark max 58.1325 versus MLX 60.9485 tok/s (−4.6202%), 10 measured each. Nine next choices agreed, only 1/3 longer greedy outputs exact; normalized score RMSE .47715. Source/weight parity did not imply numerical parity. | C3; `1e0be01f` |

## 8. Matched-checkpoint arithmetic repair and optimized kernels

All rows in this section keep the official checkpoint's stored bytes and
quantization fixed, without speculation. The implementation is committed as
`74e6c5c2d612f58555e8a85a2013dad23f02d9d7`. Multiple candidates accumulated
correctness work before replacing slower compatibility paths. Stock-stage
hashes use matching reference semantics; old controls remain unchanged.

| ID | Attempt | Outcome and decision | Evidence/source |
| --- | --- | --- | --- |
| M01 | Explicit stock HC/PLE BF16 primitive boundaries | **Retained, correctness:** BF16 normalization/gating/residual boundaries, primitive sigmoid and mean behavior. Initial reference-heavy `hc-ple-rounding-v1`/`hc-ple-mean-v2` maximums 41.3841/41.2751 tok/s, logprob RMSE about .51355. These were numerically motivated diagnostics and slower than the original adapter. | C4; A16 |
| M02 | Affine Q8/group64 KV encoding/dequantization | **Retained, correctness:** BF16 scale/bias, codes derived from unrounded scale, one final BF16 dequant round. Product-first/two-round variant failed stock oracle and was corrected. Stored weight quantization unaffected; cache arithmetic explicitly changed. | C4; final source/provenance |
| M03 | Ordinary RMS BF16 normalize-then-gain | **Retained, correctness:** stock primitive rounding; F32 gain promotes F32 and BF16-only wrapper refuses it. Existing alternate descriptors retain their route. | C4 |
| M04 | Primitive attention sigmoid, cached output norm, fused Q/K and QSA entries | **Retained, correctness:** all four entry routes covered. Attempt to compile an entire shared source with Safe mode changed strict dense parity; original F32 attention dot reader retained. No blanket math-mode change. | C4 |
| M05 | Stock-compatible GDN convolution/beta/norm/gate arithmetic | **Retained, correctness:** reference boundaries added before optimized recurrence. `gdn-compat-v1` 41.0558 tok/s max and RMSE .563719 are separate diagnostics, not isolated benefits. | C4 |
| M06 | Custom BF16 sigmoid helper | First candidate differed at −6.84375; **corrected** integer RNE/F32-exp helper matches all 65,280 finite BF16 inputs against custom reference. Primitive LUT is distinct: it matches stock primitive table despite three CPU-subnormal differences. | C4; source-bound gate tests |
| M07 | BF16 MoE router/softmax/top-k/weighted output boundaries | **Retained, correctness:** stock primitive sigmoid and score-weight order before rounding. `moe-compat-v1` max 42.2754, RMSE .600043; not a speed-only optimization. | C4 |
| M08 | HC tail fusion v1 | **Rejected reference assumption:** generic reference GEMV was not the stock skinny/register-A path; original test did not establish correct stock equivalence. | C4; preserved private draft |
| M09 | HC tail fusion v2 | Corrected actual stock controls; retained as an intermediate compatibility implementation. `hc-tail-fusion-v1` diagnostic max 42.9764, RMSE .603272. Later complete HC read superseded the reference-heavy dispatch structure. | C4 |
| M10 | Complete stock-order HC read under platform-default math | **Rejected:** F32 partial-stage hashes failed, even where coarse final checks could hide differences. Failed source snapshot preserved; goldens not weakened. | C4; A17 |
| M11 | Complete HC read with scoped Safe Metal 3.1 | **Retained, optimized implementation:** exact 16 stage bindings/four initial stock tests, later seven controls/192 stock bindings. Two-dispatch ND/U2V route replaced slow compatibility operations. `hc-full-safe-v1` max 66.8420, RMSE .500866; separate window, no single-kernel causal percentage. Cache keys include math mode; default API callers unchanged. | C4; A18 |
| M12 | GDN F32 ValueKey state layout and 192-thread stock decode | **Retained, optimized implementation:** transpose/access policy `[H,V,K]`, old KeyValue fallback preserved. v1/v2 test fixtures transposed history incorrectly; v3 repaired fixtures before admission. `gdn-value-key-v1` max 67.3696, same .500866 RMSE. This early peak is not final cross-engine benchmark. | C4 |
| M13 | Official prompt N−1 prefill + final S1 holdout | **Retained, correctness:** aligns served final-prompt primitive geometry; usable state fence clears only if both phases succeed. Manual holdout/failure tests pass. `official-holdout-v1` max 64.9705, RMSE .471823; numerical work can cost speed. | C4 |
| M14 | Native F32 head sum then BF16 cast (v2) | **Rejected:** two stock head fixture hashes failed. Same storage dtype with different reduction tree was insufficient. | C4; A19 |
| M15 | Stock Q4/group64 vocabulary QMV, v3/v4 | **Retained, correctness/optimized path:** actual stock r4 QMV order under Safe Metal4; BF16 logical logits widened into F32 buffer. v4 adds signed-address guard and final SIMD-width32 fallback. Four controls, stock hashes at widths128/256; full-vocabulary stock hash remains unqualified. `head-stock-qmv-v4b` max 66.6709, RMSE .474325. All six legacy head callers retain false/default routes. | C4 |
| M16 | Truthful head precision metadata | Initial artifact guard refused edited legacy “F32” policy before inference. **Corrected tooling:** separate F32-storage/BF16-logical fields and schema-specific allowlist, source/binary unchanged for that repair; not an inference gain. | C4; source receipts |
| M17 | TB12/DB16 pipelined ValueKey GDN prefill | **Retained, bounded optimized implementation:** guarded long, aligned, no-capture route under Safe3.1; seven dense recurrence fixtures plus sparse chain exact. Initial complete dense-chain failure occurred upstream in Q/K preparation, preserved separately. `gdn-pipelined-prefill-v1` max65.9134, RMSE .451247. | C4 |
| M18 | Contiguous four-values-per-lane Q/K reduction | **Retained, correctness:** old strided tree differed from stock at dense S65/S129 preparation; correction makes full chain state/history/gates/norm/probes/repeats exact for those fixtures. `gdn-qk-contiguous-v1` max65.4959, RMSE .428863. Does not prove every prefill shape. | C4; A20/A21 |
| M19 | Stock-order Q8 GDN projection kernels | **Retained, scoped implementation:** actual joined 16,480×2,560 and output 2,560×6,144 shapes, Safe4, strict dtype/group/SIMD32 guards; fallback for other descriptors. Three controls/two actual-shape stock hashes exact. `gdn-q8-stock-v2` max65.5894, RMSE .461196 in its earlier window; no reliable isolated decode gain. | C4 |
| M20 | HC WR1 attention-write→MLP-read pair | **Retained, exact scoped fusion:** disjoint 20 KiB scratch, canonical MoE copyback; seven controls/192 stage bindings, four fixtures×32 frames. 256-sample hash and nine numerical summaries exactly equal immediate predecessor. Separate max65.4959→65.7951 (~+.46%) is **marginal/inconclusive**, not an isolated reliable speed claim. | C4 |
| M21 | Non-dyadic affine4 MoE dot/compiler controls | **Correctness-only investigation:** compact E10 actual three-stage stock goldens match in three compiler modes (9/9). No evidence justified another global compiler-mode change; none adopted. Compact fixture does not qualify E512 host admission. | C4 |
| M22 | Detailed Metal stage profiler | **Retained diagnostic:** intermediate `head-detailed-profile-v1`, profile-disabled max66.1530. Nonwarm profile has 32 passes (held prompt S1 +31 generated), GDN36.39%, MoE28.42%, HC14.96%, QSA13.21%, head6.38%. Inclusive groups overlap; profiling perturbs scheduling and predates later changes. Initial 31/32-pass parser assumption corrected; WR1 changes marker count to 582−P within1024 slots. | C4 |
| M23 | Final combined-source correctness/regression seal | 797 source files match implementation commit; 683 native tests pass/46 ignored, vendor69/35 ignored, packaging20. Three lifecycle and one holdout/failure controls pass; retained Lily 4K/256 hash exact on final worker. **Correctness evidence**, not new throughput. | C4 |
| M24 | Final maximum-throughput ABBA | **Validated engine comparison:** Stark66.8075265 versus MLX61.1710953, +9.2142068%, 10 measured each, identical checkpoint/quantization/no speculation. Stark57.24–66.81, MLX55.65–61.17. Distinct internal timers and MLX unused-final-forward cost retained. This qualifies the named maximum, not a floor or whole-model quality. | C4; `3d22b4df`; A22 |
| M25 | Independent aggregate auditor v1/v2 | v1 **failed** new factory policy-digest serialization (compact versus pretty JSON+newline). v2 uses schema-specific canonical serialization, no evidence rewrite or relaxed binding; 54 offline controls and389 report checks pass. Arithmetic/provenance audit is not numerical equivalence. | C4; A23 |
| M26 | Locate first longer greedy divergence | v1 preparation/environment guard gaps fixed with69 mocks; v2 finds first difference at output11 after10 mapped common IDs. Ordinary/instrumented MLX text/count/finish match, but fresh common-prefix4106-token replay changes MLX's own choice. **Unresolved diagnosis:** no causal attribution, margin interpretation, near-tie acceptance or quality pass. | C4; A24 |

The final candidate's nine aligned next-token choices agree with MLX, but
pairwise rank agreement is 970/1,068 and log-probability RMSE .428863.
Only **1/3** longer greedy decoded outputs match exactly. The optimization
bundle beats the named reference on maximum throughput while leaving these
numerical differences open. Neither stock component hashes nor nine argmax
matches erase the failed longer controls.

The complete three-measured-run sequence is retained in C4. In addition to the
rows above, `attention-compat-v1` max42.6532 and
`final-official-fast-v1` max66.1726 are standalone engineering windows.
They are not paired to each other or to the final MLX ABBA. The final engine
comparison, rather than subtraction between these windows, supplies the
current performance claim.

## 9. Related all-ref work explicitly outside the no-speculation result

Reachable Git history also contains an independent Lily MTP series. It is
recorded here to avoid losing useful experiments or confusing its 78–83 tok/s
figures with this task's no-speculation result. These commits are not part of
the matched-official implementation above. MTP remains disabled in every
comparison claimed by this ledger's primary result.

| ID | Related attempt | Recorded outcome and scope |
| --- | --- | --- |
| X01 | Exact rejection MTP and batched verification | `495f2ea3`, `1586c3f8`, `6dbda328`: actual filtered draft distribution, acceptance/residual resampling, rollback and target bonus; different seeded route trajectory allowed. Separate branch evidence, not an authorized substitute for no-speculation throughput. |
| X02 | MTP profiler and command-lifetime correction | `c4a5b008`, `a7c9e10c`: diagnostic verifier profile plus ownership fix. Profile-disabled short window +5.30% on/off; no attribution to profiling itself. |
| X03 | Batched exact sparse QSA | `8bec97e7`: up to four independent causal queries, 96 dispatches/barriers removed at three verification rows; 140 complete target readouts exact. Short on/off window68.1585→79.8312 pooled. |
| X04 | Batched exact Q4 GDN projection/convolution/scatter | `a1a2a5f4`:108 dispatches removed across36 GDN layers; numerical/rollback and140-readout controls exact. Short on/off66.6554→80.4283. These separate windows do not isolate each source change. |
| X05 | Expert union-v3 | **Shelved:**768 projection/eight complete-MoE component cases exact; full-overlap synthetic1.018×, partial.879×, disjoint.672×. No model generation gain, production unchanged. |
| X06 | Longer MTP controls | `a060ff31`:16-pair top-k off67.4633/on78.3329 (+16.11%,16/16 paired wins), only8/16 on requests≥77; nucleus8-pair70.5978→71.0097 (+.58%,3/8 wins), median wall slower. Does not establish77–80 across modes or a floor. |

The [preserved MTP report](https://github.com/jshahbazi/stark/blob/a060ff31500f2338fdff7c309e3081f833ecdf73/docs/benchmarks/QWEN38_MTP_OPTIMIZATION_2026_10_30.md)
contains its separate source/worker identities, controls and licenses. Its
figures are not included in the website's same-checkpoint no-speculation claim.

## 10. Supplemental aggregate identities

Private relative names identify evidence, not distributable files. All new
entries below are aggregates or test logs with scalar results/hashes; model
weights, prompts, generated text, token vectors and raw probabilities are not
included. The original study tables remain the primary identity catalog for
earlier experiments. Hashes of deleted rebuildable binaries identify historical
runs without claiming those binary files still exist.

| ID | Private aggregate or source identity | SHA-256 |
| --- | --- | --- |
| A01 | `uzu-batched-top20-32-v1.json` | `5832a156745b226f9253a6c142bf0301e660106cbee455224ec1f856ec2e529f` |
| A02 | `uzu-parallel-router-top20-32-v1.json` | `280677ba461d9df174c1401df1e91912e88d51ec6b07a80376f4d6241a2155a9` |
| A03 | `uzu-parallel-router-top20-4k-v1.json` | `0af6b65815e8920f8eb550ca30b314eebca566b7eb3dc3d12ea4971be6bfff70` |
| A04 | `uzu-f32-partials-top20-4k-v1.json` | `3bea5c17c98e11348b672468d30c35ffcd5aba15ad30e23b9dc70f0b6c333eec` |
| A05 | `uzu-lexical-router-top20-4k-v1.json` | `b5930a348d54d08037468b5065fc9f2403df05ec93235ad4abd807000e989f32` |
| A06 | `uzu-selective-gemv-mask5-same-loaded-32-v1.json` | `4e09d2b02f05f8a1e790a748ce58cf6fd1bc68fe6fd210d8a6c5e5db3aa20bb4` |
| A07 | `full-vocab-control/independent-affine-controls-v1.log` | `d7fcfdfd40de1360cb24c6713165e365e3c612a4e99ff1eb9b4b5415916e8b2f` |
| A08 | `full-vocab-control/local-ilp-components-actual-q4-v2.log` | `41d319809f3249e2a04c91f17d88aa860623aec634b4ba8b856ac3665e2f0dcb` |
| A09 | `full-vocab-control/group-ilp-components-actual-q4-v2.log` | `a99356c17070aed826510febc930639339062e44cfbd90bd6db0692044904bec` |
| A10 | `full-vocab-control/wide-rowwise-components-v1.log` | `11cdcaac4e6fb1609426ef576403f39230bff888b18ae85f10c36470f520f762` |
| A11 | `full-vocab-control/wide-row-local4-components-v1.log` | `51ebf6b130d3970802a884f1abfb436029ca252e0863ab10bc179bac59382d7e` |
| A12 | `full-vocab-control/affine-layout-components-v1.log` | `5c0ac9b65d7a7ead53d6b98795273551bffd9ccecab3502cd2b761b529852264` |
| A13 | Original `full-vocab-control/gdn-float2-controls-v1.log` failed exact gate | `43b24c6a5f403837f8886d5412ea93d7cb83cb55d101eb0d62cd1ac731e8b836` |
| A14 | `full-vocab-control/gdn-scalar-pair-controls-v1.log` diagnostic | `e8e345e1598f70014e23cec34995a2e1bb99a26303d7325443513689453be7ef` |
| A15 | `full-vocab-control/gdn-v3-exploratory-components-v2.log` | `ebd1167a954413d243a2c197abad14029d2a94a97c23ac33875f784ea6622222` |
| A16 | Current final source manifest,797 frozen files | `a5135b3a0e0222730a86b999ae91411a3bd2ff1b17003a588e4e19cdb2666921` |
| A17 | Failed platform-default HC source manifest | `6afeb81e13d1d51afa6cb50edac2b42e44a40f0c0b2e7b58817c87ce8acbcc21` |
| A18 | Final complete source-tree manifest | `301a5b1f92901035edc78678cbc14ed1b8c1d96aacc2ee1cdddcf337c2ad7cc2` |
| A19 | Failed native-sum head source manifest | `0d773ade0ba18d53a1e564db764271d4fd87a5e4b2c26cb42d84d7d6fd63cdce` |
| A20 | Failed dense whole-chain diagnostic | `6d217cef41a9f12f87135abc3f07f73270124e32b3421786fd8fd5fc6cb85830` |
| A21 | Corrected contiguous-Q/K complete-chain report | `37636394eabaeb25ddb1d6c9d4d7cf9cfc881bd679b3cf611893108aaa78fca2` |
| A22 | `matched-final-sampled-abba-4k-v3.json` | `4fc0ba81171a8c0f96400a41a9da80b053ad57b97a5c4f5ad462869f88163f43` |
| A23 | Aggregate auditor v2 source | `e1fd1ea1aff816037f3bc796856c9f3a7a9cb0e42a4e5a8a9778d3dfe48ba3e5` |
| A24 | `final-first-greedy-divergence-v2.json` | `e856037c5b0ad3adb227133755a34c68cc50c14526a889ffbee1c8c3b1582fbe` |
| A25 | Final generation worker | `3946afcf18b9a024b93ee05dfa55952e468a9038dce8fc0a9c2e5b4e660a7a05` |
| A26 | Final scoring worker | `f778948e8e40caa7add367c172ff75fc7914c4fbe59f5e71fd46b2aee65d4305` |
| A27 | Shared checkpoint manifest | `11246ebbea8a24a988517ebeb59623b857df89120e5ae3becd48807194542ef6` |
| A28 | Complete shared-checkpoint payload verification | `e112599b932abf197011afe5eb91bc1f8a2bb55716eda470b6ee4ef0fcb275e8` |

## 11. Git lineage and coverage limits

The original mixed-precision/sampling/kernel/stability/power sequence was moved
after its earlier branch merged. Commit pairs below represent equivalent
history movement, not additional independent experiments:

| Original | Active-lineage copy | Subject |
| --- | --- | --- |
| `8dc6ccd0` | `0bb1df44` | Mixed precision/native acceleration |
| `d13e5cde` | `80474e05` | Sampled generation/BF16 profile |
| `908ae21f` | `eba4cafe` | HC cache/GDN fusion |
| `9204b0b9` | `174d35ef` | Exact sampler/host contention |
| `6f953e8c` | `565a9a5f` | AC power controls |

The full-Uzu work is preserved on `codex/qwen38-uzu-execution` at `17485499`;
the subsequent fast/matched work is on `codex/qwen38-mlx-benchmark` through
`3d22b4df`. A branch name alone is not a source identity; use the recorded
commits, manifests and worker hashes when reproducing a run.

This compilation checked the 2026-10-29/30 conversation's recorded decisions
and targeted original session history, dated Qwen studies, reachable source
commits, preserved full-Uzu documentation, private component README/aggregate
records and the final matched evidence bundle. It records identifiable
algorithm/kernel variants, abandoned hypotheses and material tooling failures.
Routine formatting retries, every download-status poll and mechanically
equivalent rebuild are not separate optimization attempts.

There are explicit remaining gaps. Prepared-only proposals may have no execution
receipt; deleted temporary binaries cannot be inspected anew; older working-tree
test logs sometimes lack immutable source manifests. Some component timing has
large ordering effects. The full standard-Uzu path still requires refreshed
final-source conformance/numerical/long-context qualification. Cross-engine
longer-generation numerical differences remain unresolved. GLM follow-up was
requested after Qwen, but no completed full-Uzu GLM migration or matched GLM
throughput claim is supported by this Qwen experiment chain.

Further optimization should keep the same official checkpoint and no-speculation
controls, use the final matched baseline, and record new source-bound experiments
separately. The retained intermediate profiler suggests GDN and MoE are the
largest areas, but it predates the final changes and must be refreshed before
using its shares to predict an end-to-end improvement.
