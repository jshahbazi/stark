# OLMo 3 and Gemma 4: findings and experiment history

Report date: 2026-11-08. Branch: `codex/native-olmo3-gemma4`. Latest engine implementation reviewed: `5aeaab96cae9354ef95eda1cd070fb2c51da11fa`; latest benchmark publication reviewed: `294b228663c3bf47c6ee250aa4083cb42c060aa3`.

Native OLMo 3 7B text and Gemma 4 E2B text/image execution were delivered on this branch. Both pinned Q4 text checkpoints now pass their numerical gates and bounded end-to-end checks. The largest measured speedup came from parallel Q/K normalization: 19.46% lower paired request latency for OLMo and 3.07% for Gemma versus the precision-matched C2 baseline. Later source RMS reuse and Gemma residual fusions improved Gemma's paired work rate another 2.57% over C3, closing its remaining MLX gap on the fixed workload. OLMo received no material speedup from that later bundle.

This report backfills the whole chat, from family selection through the accepted reruns after the Stark desktop app was closed. It reconciles conversational progress reports with committed validation records and aggregates. Early, superseded and invalid results remain visible. It adds no new runtime measurements and does not reclassify older failed runs as passes. The [model experiment register and template](../../MODEL_EXPERIMENTS.md) define how to continue this history.

## What is qualified now

| Model / mode | Correctness and execution evidence | Latest performance finding | Remaining boundary |
| --- | --- | --- | --- |
| OLMo 3 7B Instruct BF16 text | Five canonical text cases; generation/scoring replay, restart and recovery | No qualified BF16 speed comparison in this investigation | Pinned checkpoint and fixtures only |
| OLMo 3 7B affine W4/group-64 text | 14/14 numerical gates and greedy matches; 13/14 full-vocabulary exact cases; generation/scoring lifecycle checks | C3 Q/K optimization: +24.16% paired work rate versus C2. Latest C4: -0.05% versus C3, +2.31% versus MLX | The 4,101-token case remains non-exact within its unchanged gate |
| Gemma 4 E2B BF16 text | Five canonical text cases; worker lifecycle checks | No qualified BF16 speed comparison here | Canonical HF arithmetic is a separate profile from MLX Q4 |
| Gemma 4 E2B affine W4/group-64 text | 14/14 full-vocabulary exact cases; 129 cached forwards with all 4,902 component hashes matching; worker lifecycle checks | C3: +3.16% paired work rate versus C2. C4: +2.57% versus C3; measured +0.44% versus MLX, reported as MLX parity | Bundle-level gain; individual fusion contributions were not isolated |
| Gemma 4 E2B BF16 image artifact | Three canonical image cases; preprocessing exact on fixtures; worker recovery and two cold/cached client checks | No qualified image performance result | Separate BF16 vision artifact; installed desktop UI execution remains unmeasured |

The latest fresh qualification records 717 native CPU/Metal tests and 486 CPU/grammar checks (476 unit, seven integration, three doctests), all passing, plus 28 Q4 cases, ten canonical text cases, three canonical image cases and worker/client lifecycle checks. These counts describe the [source-fusion qualification](2026-11-02-olmo3-gemma4-source-fusions.md). Earlier suites had different counts. The converter suite was unchanged at this final stage; its earlier 5,065-pass/90-skip evidence was retained rather than rerun.

## Evidence and revision map

| Phase | Implementation / publication | Authoritative record |
| --- | --- | --- |
| Initial native support | `0f3e855f` / `9257d80a` | [Native validation](2026-10-28-olmo3-gemma4-native.md) and [initial diagnostic timing](../../benchmarks/OLMO3_GEMMA4_NATIVE_MATCHED.md) |
| Broad Q4 source-precision repair | `b53320a1` | [Precision investigation](2026-10-28-olmo3-gemma4-q4-precision.md), including failed intermediate controls |
| Gemma cached-trajectory repair, C2 | `2e96ec86` | [Cached trajectory completion](2026-10-28-olmo3-gemma4-q4-precision.md#gemma-cached-trajectory-completion) |
| Parallel Q/K normalization, C3 | `598b3f6e` / `3394cbe7` | [Repaired Q4 benchmark](../../benchmarks/OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md), [aggregate](../../benchmarks/olmo3-gemma4-q4-repaired-2026-10-29.json) |
| Prepared residual fusions | Then-uncommitted candidate based on `3394cbe7` | [October checkpoint](2026-10-29-gemma4-source-residual-fusions.md); timing/push were pending at that checkpoint |
| Fresh qualification and accepted reruns, C4 | `5aeaab96` / `294b2286` | [Fresh validation](2026-11-02-olmo3-gemma4-source-fusions.md), [current benchmark](../../benchmarks/OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md), [aggregate](../../benchmarks/olmo3-gemma4-source-fusions-2026-11-02.json) |

The dates above are the existing source-bound record labels. Their historical pending statements describe those checkpoints; follow-up records establish completion. Raw fixtures, captures, binaries and candidate patches remain private under ignored `runtime/olmo3-gemma4/`. The linked records publish their receipt hashes. This retrospective does not require publishing the chat or private runtime artifacts.

## Scope, sources and architecture

| Profile | Repository | Immutable source revision |
| --- | --- | --- |
| OLMo BF16 | `allenai/Olmo-3-7B-Instruct` | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` |
| Gemma BF16 text and separate vision conversion | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| OLMo Q4 | `mlx-community/Olmo-3-7B-Instruct-4bit` | `d732c91ae02e90cd5d86e810fbeb9741794b4dd9` |
| Gemma Q4 | `mlx-community/gemma-4-e2b-it-4bit` | `238767527555cb75a05732a84dff5d6ba0dd6809` |

The Q4 artifacts keep packed affine W4/group-64 matrices, BF16 activations/KV and phase-specific source accumulation/materialization. This is native Rust/Metal execution; MLX supplies a pinned external reference and inspected arithmetic source, not a runtime dependency. Text references use MLX 0.32.3 and MLX-LM `1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7`. Canonical HF image and separate MLX-VLM image references have distinct locks and policies. Their results cannot be pooled as one oracle.

OLMo's tested geometry has 32 query and 32 KV heads with D128, full-width Q/K RMS normalization, 24 sliding-window layers and eight full-attention layers with configured YaRN. Gemma E2B has 35 layers, eight query heads and one KV head in producing layers, D256 local/D512 global attention, shared KV consumers, proportional RoPE, PLE gates/scalars and a final softcap. These differences explain why several optimizations have different value across the two models.

### Family-selection research

The initial advisory work also considered GPT-OSS, Nemotron, Jamba 2, Ministral/Devstral, Granite hybrid and Phi Flash. It separately suggested checking Granite 4.1 compatibility against the existing dense adapter; this was distinct from adding a hybrid runtime. The suggested implementation order was OLMo 3, GPT-OSS 20B, Gemma 4, Ministral/Devstral, then Nemotron. Existing adapter names were checked separately from native generation/scoring support. These were architecture candidates; this chat did not implement or qualify the other families. Flint, Qwen3.8-Flash-Next and GLM-5.3 were already separate branch work. No support claim for any of those families follows from this OLMo/Gemma report.

The [Mac engine evidence survey](../../benchmarks/OLMO3_GEMMA4_MAC_ENGINES.md) found no defensible globally fastest engine for either whole family. Its strongest controlled Gemma E2B comparison was a publisher's M5 Pro study: BaseRT 149.8, MLX-LM 146.9 and llama.cpp 103.3 decode tokens/s. The quantizations had nominally matched bit widths, not identical bytes. OLMo reports lacked a comparable exact-checkpoint multi-engine control. mlxcel measurements involved other sizes/checkpoints or hardware. These remain historical external observations, not Stark measurements or current engine rankings.

BaseRT's runtime and Metal kernels were unavailable as source at the inspected revisions; no kernel was copied. MLX/MLX-LM and mlxcel source were inspected with pinned revisions and license boundaries. Adapted arithmetic and the Qwen-inspired scheduling idea are documented in [upstream provenance](../../../provenance/UPSTREAM.md). Metal 4 prefill and speculative/MTP reports supplied research directions, with no native performance result for these model profiles in this chat.

## Native support and initial qualification

Evidence: [initial validation](2026-10-28-olmo3-gemma4-native.md) and [current feature scope](../../OLMO3_GEMMA4.md).

| ID | Change / model | Observation | Disposition |
| --- | --- | --- | --- |
| OG-001 | OLMo native decoder and conversion | Implemented full-width Q/K normalization, sliding/full attention, YaRN and post-norm residual layout. Official BF16 text passes five cases and bounded generation/scoring lifecycle checks. | Retained; correctness qualified within named fixtures |
| OG-002 | Gemma native PLE decoder and conversion | Implemented local/global dimensions, proportional RoPE, shared KV ownership, model-specific norms, PLE gates/layer scaling and softcap. Official BF16 text passes five cases and workers. | Retained; E2B qualified, broader text layouts need their own evidence |
| OG-003 | Gemma BF16 native images | Added official processor recipe, learned vision tower/projector, image/text fusion, schema-4 admission and worker/client integration. Patch tensors match; feature relative RMS is 1.17%, 1.33%, 1.50% under the 2% gate; three greedy comparisons pass. | Retained; bounded image correctness/lifecycle qualified |
| OG-004 | Initial BF16 parity failures and calibration | Earlier OLMo and Gemma profiles each had failures. Final profiles were calibrated against independent HF CPU BF16/F32 controls; a mismatched-context negative control fails. Final five-case suites pass. | Earlier failures preserved; final calibrated profiles retained |
| OG-005 | Initial affine Q4 conversion/execution | Both convert and run, but OLMo passes only 4/5 numerical cases and Gemma 3/5. All ten greedy choices and short workers match, while longer generated trajectories diverge. | Failed full numerical qualification; superseded by OG-006 through OG-025 |

For OG-004, gate calibration is part of the original BF16 investigation; it must be disclosed rather than described as an unchanged protocol throughout the chat. Once those acceptance profiles were established, the subsequent Q4 repair retained their gates. OLMo's initial Q4 one-token raw RMSE was 0.127956 against 0.1, and centered RMSE 0.047431 against its case bound 0.047090. Gemma's 23- and 129-token Q4 cases failed. Greedy agreement alone concealed these failures.

The recorded OLMo profile is `bf16-centered-logits-v3`; Gemma uses `gemma4-canonical-bf16-v4`. Both cap centered maximum absolute error and absolute mean error at 0.75, and reference-to-native KL at 0.01. OLMo additionally caps raw RMSE at 0.1 and centered RMSE at `1/128 + 0.02 × reference centered RMS`. Gemma caps centered RMSE at 0.15 and `1/128 + 0.08 × reference centered RMS`. The initial Gemma calibration controls reached 5.405% centered-relative L2 for HF CPU BF16/F32 and 6.068% for HF BF16 versus the separate MLX profile. Those controls explain the calibration; they do not replace the stock-MLX Q4 oracle. No separate image-specific BF16/F32 logit control was run; image logits use the recorded text criterion.

The image path is a separate official BF16 artifact, not a Q4 vision qualification. It accepts one bounded PNG/JPEG, up to 2,520 patches and 280 pooled soft tokens. E4B images, other Gemma architectures, audio and video are outside the qualified scope. Installed desktop integration is wired, but installed UI execution and image throughput were not measured.

## Precision repairs that worked

Evidence for OG-006 through OG-025: the [precision investigation](2026-10-28-olmo3-gemma4-q4-precision.md), including its operation-level controls, routing bounds and immutable receipts. These changes were qualified together; their isolated controls do not assign an individual end-to-end speedup.

| ID | Model / change | Failure isolated and successful repair | Disposition |
| --- | --- | --- | --- |
| OG-006 | Both: affine GEMV bias correction | Source sums each four-input group through BF16 additions before its F32 affine bias contribution. Reproducing that boundary removes all 92 disagreements in a 128-row OLMo projection control. F32 dot accumulation remains. | Retained in explicit source policy |
| OG-007 | Both: M=1 expression schedule | Match source eight-value lane form, four-product subtotals and F32 expression fold/dequantization schedule. Source-compatible scalar storage alone was insufficient. | Retained |
| OG-008 | Both: prefill QMM scheduling | Preserve original logical projection widths, source matrix/vector thresholds, BF16 partial scratch and typed reduction, while retaining physical packed strides. Actual Gemma QKV controls become exact at M18, M128 and M511. | Retained; QMM and dense-attention split-K use distinct precision policies |
| OG-009 | OLMo: SwiGLU sigmoid | Materialize source exponential, denominator, reciprocal and sign-dependent result in BF16. Original expression differs on 4,682/11,008 gate values; complete source expression has zero differences in that control. | Retained |
| OG-010 | Both: RMS normalization/learned scale boundary | Normalize/statistically accumulate in F32, round normalized value to BF16, then apply learned gain. `OnlyNormalization` matches stock MLX where `FullLayer` did not. Canonical HF remains a distinct profile. | Retained in source profile |
| OG-011 | Gemma: quantized embedding | Round dequantized embedding to BF16 before applying the BF16 embedding scale. Scaling a wider dequantized value first changes the result. | Retained |
| OG-012 | Gemma: softcap | Preserve separate BF16 division, tanh and multiplication, rather than a wider fused expression. | Retained as part of complete source repair |
| OG-013 | Gemma: approximate GELU and PLE scalars | Follow source weak BF16 coefficients/intermediates and profile-specific PLE scalar boundaries. Validate actual GPU gate inputs in addition to synthetic controls. | Retained; canonical HF opmath remains separate |
| OG-014 | Both: source RMS reduction tree | Use source four-contiguous-value ownership/reduction and reciprocal square root. Canonical native can be closer to an F64 control yet differ from MLX at BF16 midpoints. | Retained behind default-false source metadata; row limit 4,096 |
| OG-015 | Gemma: value normalization inheritance | Propagate source reduction/upcast settings from K normalization to runtime-created V normalization. Removes five differences across 261,632 actual values. | Retained |
| OG-016 | OLMo: attention score scaling | Scale completed F32 QK scores instead of scaling BF16 Q before the dot product. Actual layer-0 control becomes exact across 200,704 outputs. | Retained source-only specialization |
| OG-017 | OLMo: online softmax reduction/finish | Preserve local pairing/shuffle/fold order and rounded F32 reciprocal-then-multiply finish. Closes later one-value layer-1/layer-11 discrepancies. | Retained; explicit source expression/compiler schedule |
| OG-018 | Gemma: materialized attention and softmax | Materialize BF16 QK scores/probabilities before V product and use source global-max/reduction ordering. Eliminates 686,908 output differences in the layer-0 511-row control; later probability differences also close. | Retained for bounded causal unit-scale BF16 D256/D512 source shapes |
| OG-019 | Gemma: populated-cache D512 dispatch | Source rejects fused D512 vector attention below 1,024 keys. Match materialized attention there; D256 keeps its vector route. The actual cached control becomes exact across 4,096 outputs. | Retained; routing boundaries tested |
| OG-020 | Both: precise source vector-attention compile policy | A source-only Safe/contraction-enabled kernel closes isolated cached BF16 disagreements caused by relaxed arithmetic. Legacy routing remains separately controlled. | Retained for audited flat single-query shapes |
| OG-021 | Gemma: short wide-head prefill QK | Source two-way dense split-K uses F32 partials/reduction then one BF16 score store. Fixes the remaining 23-token QK boundary; output batching alone did not. | Retained for bounded D512/eight-Q/one-KV source predicate |
| OG-022 | Both: GPU rotary coefficients | Match GPU frequency/trigonometric operations and explicitly call `metal::precise::pow`; a Safe pragma after Metal headers cannot undo fast-math preprocessor selection. Gemma's 23 differing frequency coefficients become zero. | Retained for explicit supported source RoPE configurations |
| OG-023 | OLMo: YaRN prescaling | Materialize source weak BF16 attention factor before rotation. Removes remaining 11 Q and six K differences in the isolated YaRN control. | Retained |
| OG-024 | Gemma: rotating-cache physical traversal | Equal visible KV values had different physical reduction order after window wrap. Map source traversal onto existing native storage without copying KV. First trajectory divergence at input position 512 closes. | Retained in C2 cached-trajectory repair |
| OG-025 | Gemma: cached dense-wide QK | Later position 519 exposed source wide-vector QK dispatch/reduction differences. Match the bounded source route rather than assuming mathematically equivalent QK batching is identical. | Retained in C2; final trajectory matches all 4,902 component hashes |

The first broad Q4 repair already passed both 14-case gate suites. That was insufficient to establish a matching 128-token Gemma stream: its sequence cases included non-exact results, and the longer cached trajectory exposed OG-024/025. After those fixes, Gemma matches all 14 full-vocabulary cases and every recorded trajectory component. OLMo remains exact in 13/14 cases; its 4,101-token input has raw RMSE 0.0971913874, centered RMSE 0.0358055215, centered maximum error 0.184105578 and KL 0.00000172058, passing the existing bounds. No universal bit-exactness claim follows.

### Precision hypotheses that failed or were insufficient

| ID | Model / attempted change | Observed result | Decision |
| --- | --- | --- | --- |
| OG-026 | OLMo: packed-weight corruption hypothesis | Audited QKV/MLP codes, scales and biases exactly match source. | Rejected as explanation in the bounded audit |
| OG-027 | Shared dequantization: blanket BF16 rounding | Original wide and typed scalar expressions already match 131,072 source values; blanket BF16 rounding introduces 76,800 differences. | Rejected; round only at demonstrated source boundaries |
| OG-028 | OLMo: change F32 fold alone | Leaves all 92 disagreements in the actual projection control. | Insufficient; OG-006/007 required |
| OG-029 | OLMo: sigmoid-only / RMS-plus-sigmoid intermediates | Pass 4/5 and 3/5 respectively. Individual exact primitives did not fix whole-model disagreement. | Not qualifying candidates |
| OG-030 | Gemma: embedding-only and embedding-plus-softcap intermediates | Each passes only 2/5 stock-MLX cases, despite 5/5 against canonical HF controls. | Insufficient for stock-MLX Q4 qualification |
| OG-031 | Gemma: scalar/profile adjustments alone | MLX-profile candidate passes 2/5 stock cases; canonical-scalar MLX control 4/5; HF-opmath control 5/5. | Keep oracle-specific evidence separate; no stock pass |
| OG-032 | Gemma: uniformly F32 QMM partial scratch | Only 2/5 original and 3/9 sequence cases pass. | Rejected; more F32 storage did not reproduce source QMM |
| OG-033 | Both: source-aligned prefill without complete cached-step repair | OLMo 2/5 original and 6/9 sequence; Gemma 2/5 and 4/9. Gemma's full 511-row prefill can be exact while the next cached step fails. | Insufficient; populated-cache controls required |
| OG-034 | Source RMS: reciprocal-square-root substitution alone | No effect on the isolated disagreement. | Rejected as explanation; reduction tree mattered |
| OG-035 | OLMo attention: earlier finish/scalar-helper controls | Did not remove the layer-1 discrepancy. | Not promoted as a fix; final source reduction/finish controls kept separately |
| OG-036 | Gemma: QK output batching alone | Did not fix the contested BF16 score. | Insufficient; source split-K precision required |

The practical finding is that accumulation dtype, storage dtype, expression order, reduction tree and dispatch geometry are separate contracts. Mixed precision helped when it preserved the reference's operation boundaries. Uniformly widening intermediates or rounding every dequantized value was unsuccessful.

## Performance experiments and dispositions

Evidence: [precision/performance investigation](2026-10-28-olmo3-gemma4-q4-precision.md#performance-investigation), [C3 comparison](../../benchmarks/OLMO3_GEMMA4_Q4_REPAIRED_2026_10_29.md), [fusion checkpoint](2026-10-29-gemma4-source-residual-fusions.md), [C4 comparison](../../benchmarks/OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md).

| ID | Scope / attempt | Result | Disposition |
| --- | --- | --- | --- |
| OG-037 | Shared source Q4 GEMV tile/policy selection | Earlier 71-configuration sweep; later 16-shape/286 tile-policy controls exact within named arithmetic policy. Production includes a bounded large-M5 source dispatch table for six projection shapes. | Retained component tuning; no isolated end-to-end gain assigned |
| OG-038 | OLMo: parallel contiguous Q/K RMS | Earlier route emulated 32 logical groups in one SIMD group. Reuse the parallel source RMS kernel for each full 4,096-element Q/K row with unchanged arithmetic. Instrumented spans drop roughly 2.52 ms to 0.13 ms per forward; accepted request latency falls 19.46%, rate rises 24.16% versus C2. | Retained in C3; strongest qualified OLMo gain |
| OG-039 | Gemma: batched small-head Q/K RMS | Batch eight Q rows or one K row at D256/512; shared-KV consumers normalize Q only. Accepted request latency falls 3.07%, rate rises 3.16% versus C2. | Retained in C3 |
| OG-040 | Scale-plane transpose | No material measured end-to-end improvement. | Not promoted; private trial preserved |
| OG-041 | Direct-nibble dequantization | Kernel changes did not establish useful end-to-end improvement. | Not promoted |
| OG-042 | Full source pointer-QMV adaptation | No material end-to-end improvement; CPU subnormal disagreement was checked against actual MLX and matched its flush-to-zero pattern. | Not promoted on performance evidence |
| OG-043 | OLMo: model-owned RoPE frequency cache | Bitwise controls pass; screen improves only about 0.36%. | Not promoted; no qualified campaign benefit |
| OG-044 | Gemma: remove two cached D512 identity scatter copies | Same-input scores, probabilities and outputs exact; end-to-end screen only 0.63–0.72% faster. | Not promoted |
| OG-045 | Remove supposedly redundant KV acceptance copies | Existing benchmark state acceptance already avoids the alleged copies. | Hypothesis rejected; proposed saving absent |
| OG-046 | Private Metal 4 stage timestamps | Located OLMo's Q/K bottleneck, but observation adds substantial request cost. | Diagnostic only; removed from production/timed comparison |
| OG-047 | Both: retain four rounded RMS inputs per thread | Reuse source BF16 values through unchanged reduction/normalization; avoid second buffer read. OLMo's complete C4 bundle is -0.05% versus C3, effectively unchanged. Gemma result is bundled with OG-048/049. | Retained in C4; no separate speed attribution |
| OG-048 | Gemma: residual addition + pre-FFN RMS | Preserve BF16 residual sum before statistics while removing separate add dispatch. | Retained in C4 bundle |
| OG-049 | Gemma: PLE residual addition + layer scale | `MaterializedBF16AddScale` preserves rounded sum and final scaled BF16 store. Together OG-048/049 remove two dispatches/layer, 70 per 35-layer forward. C4 bundle improves paired rate 2.57% versus C3. | Retained; MLX parity reached on matched workload |

OG-038/039 require BF16, explicit source reduction, learned scales, `OnlyNormalization`, zero scale offset and audited geometry. Weightless V, multi-token and canonical profiles retain separate routes. OG-047 retains the source row limit of 4,096 and the existing reduction tree. OG-048/049 additionally require reference PLE; they do not fuse the canonical BF16 or legacy PLE profiles. Kernel controls cover BF16 midpoint-sensitive values, tails, broadcasts, in/out-of-place operation and sentinels; exhaustive nonfinite/subnormal equivalence was not established.

An OLMo chat update at 2026-10-29 08:24 UTC also described a "cache optimization" retaining the 128-token source output with less than 1% latency change in a quick screen. That message did not identify its patch or cache mechanism. It supplies no qualified gain and cannot establish a separate OLMo cache-traversal experiment beyond the named cache trial above. Gemma's source traversal repair (OG-024) and scatter removal (OG-044) have separate, explicit evidence.

### Accepted performance progression

All percentages below refer to complete-request work rate unless labeled latency. Comparisons are within their own campaign; percentages across dates must not be multiplied into an unmeasured cumulative gain.

| Record / build | OLMo finding | Gemma finding | Claim status |
| --- | --- | --- | --- |
| Initial native Q4 comparison | 14.0% below MLX | 2.4% below MLX | Diagnostic only: numerical failures and different long output hashes |
| C3 versus precision-matched C2 | +24.16% rate; -19.46% latency | +3.16% rate; -3.07% latency | Qualified, all four paired blocks |
| C3 versus MLX in C3 campaign | +1.55% rate | -2.15% rate, the conversational remaining 2.2% gap | Qualified historical result |
| C4 versus C3 in latest campaign | -0.05% rate; no material gain | +2.57% rate; -2.51% latency | Qualified, all four paired blocks |
| C4 versus MLX in latest campaign | +2.31% rate | +0.44% rate, reported as parity | Qualified for this workload; no general engine ranking |

The host is Apple M5 Max, 40 GPU cores, 128 GiB, macOS 26.7.1 build 25G241. Each accepted three-arm campaign has four independent loads per arm, one warmup and three measured requests per load, fresh state per request and balanced arm order. Across both families that is 24 loads, 24 warmups and 72 measured requests. Source/artifact/executable/reference hashes and all within-family output hashes agree; all existing guards pass.

The workload is pp512/tg128: 512 input tokens, 128 greedy returned tokens, 128 decode forwards and the final queued lookahead drain, native capacity 1,024. Work rate is `640 / complete-request seconds`, including setup, prefill and generation; loading/warmup are excluded. These are not standalone decode tokens/s. Paired ratios are calculated per matching block and summarized by their median; they can differ from ratios of displayed campaign-wide median times. This explains OLMo's slightly lower displayed C4 time alongside its slightly negative paired gain.

## Transfer from Qwen and GLM

The user supplied `codex/native-glm53-qwen38` for hints. The chat inspected local scheduling revision `908ae21f4894d841cf2e1ad44451643439543443` and later branch head `6f953e8ccc9934d89889a812482faa256fc5199f`. These are inspected historical identities, not a claim about today's moving branch head. [Arithmetic attribution](../../../provenance/UPSTREAM.md) and [C4 diagnostics provenance](../../benchmarks/OLMO3_GEMMA4_SOURCE_FUSIONS_2026_11_02.md#host-protocol-and-controls) retain the boundaries.

| ID | Idea | Finding for OLMo / Gemma | Disposition |
| --- | --- | --- | --- |
| OG-050 | Reuse a thread-local already-rounded value | Qwen GDN scheduling suggested retaining rounded values instead of materializing/reloading them. Applied to RMS and Gemma fusions while preserving these models' arithmetic. | Transferred; OG-047/048/049; no Lily arithmetic copied |
| OG-051 | Host telemetry helper | Reused `hybrid_host_metrics.py` with identical polling across all arms. Helped describe thermal, memory and competing-process conditions. | Used for private aggregate diagnostics; guards unchanged |
| OG-052 | Hybrid HC/GDN, sparse-attention and expert kernels | Those operators/layouts are not the dense decoder computation in the tested OLMo/Gemma checkpoints. Hybrid graph throughput gains do not predict a dense gain. | Investigated; no direct port |
| OG-053 | Large F32-to-BF16 hybrid speedups | Dense OLMo/Gemma activations were already BF16 with selected F32 accumulation. The same baseline precision conversion was unavailable. | Investigated; no equivalent gain claimed |
| OG-054 | CPU top-k/radix sampling optimizations | Current dense generation uses GPU sampling; the Qwen CPU bottleneck was absent here. | Investigated; not applied |
| OG-055 | Q8 KV cache | Potential longer-context OLMo research because it has many KV heads. Gemma E2B's one-head/shared-KV structure gives a different benefit/cost profile. | Deferred; no implementation, accuracy or speed qualification |
| OG-071 | QoS, callback-queue and power-setting changes | The inspected Qwen stability work established no reliable speed benefit from these experiments. No OLMo/Gemma benefit was measured. | Investigated; no transfer or gain claimed |

The useful transfer was a local scheduling method and measurement discipline. A larger dense decode fusion/graph may still be worth profiling, but no gain from the hybrid graph itself was established for these models.

## Invalid runs, harness mistakes and evidence limits

| ID | Attempt / affected model | What happened | Recorded resolution |
| --- | --- | --- | --- |
| OG-056 | Initial Q4 timings, both | Next-token/short-worker agreement masked numerical failures and divergent 128-token trajectories. | Kept diagnostic; repaired numerical and trajectory paths before accepted comparison |
| OG-057 | Overlapping OLMo numerical calls | Early yielded child processes briefly overlapped. | Excluded v1 runs; serialized v2 reruns completed before each next GPU job; OS lock adopted |
| OG-058 | Gemma scoring-worker fixture | One final attempt selected the canonical/other-precision worker fixture instead of the Q4 reference fixture; score gate failed. | Correct Q4 fixture rerun passes with no source/reference/tolerance change |
| OG-059 | Gemma canonical image harness | An invocation selected the text-only artifact and failed at model loading. | Correct BF16 vision artifact on a private fixture copy passes; metrics unchanged |
| OG-060 | New cached-attention dispatch unit test | Test fixture retained ring/window settings while asserting flat global routing; native run had 708 passes and one failure. | Corrected fixture; preserved failed run separately, then 709 passes; later suites expanded |
| OG-061 | Preliminary fusion timing screens | Disk scans and intermittent other GPU work caused swings larger than the target 2.2% gap; user explicitly paused timing. | No speed claim; correctness continued, frozen candidate retained for a quiet window |
| OG-062 | First resumed fusion campaigns | Stark desktop CPU bursts tripped the existing 10% background-model CPU guard. Matching outputs and nominal thermal diagnostics did not qualify timings. | Both campaigns rejected and hashed; user closed Stark; complete unchanged-guard reruns pass |
| OG-063 | Reference/harness identity changes | Freshly recomputed-prefix versus cached-trajectory protocols and cache-state-only prefill needed distinct identities. A copied obsolete summary label also required correction. | Versioned protocols/receipts preserved; summaries corrected without rewriting underlying results |
| OG-064 | Incomplete patch capture | Earlier candidate patch hash omitted then-untracked added files, despite separate source bindings. | Later complete patch includes added files, SHA-256 `9718c93efb5c78a89abd5488031672db24324b00b177b940259b0f167c105c54` |

The host guard limits each background process to 200% CPU, detected background model processes to 10%, and rejects sleep/clock discrepancies of two seconds or more. GPU work is serialized locally with an OS lock, but that lock cannot prevent unrelated applications from using the GPU. The reruns retained the guards; no application exception or threshold relaxation was introduced.

Opt-in IORegistry GPU-counter units are unverified, clients can appear/disappear between samples, and boundary intervals are excluded. Zero detected model GPU deltas, nominal thermal state and zero pageouts/swapouts do not prove exclusive GPU ownership or fixed frequency. GPU frequency was not measured. Correctness captures made during contention can retain their identity/parity value while their timings remain unusable.

One audited legacy Gemma trace changed a layer element when compiler context changed, despite default-false metadata preserving dispatch policy. Each compiled variant therefore needs its own numerical evidence; policy defaults do not establish universal binary-level equality.

## What remains worth investigating

These are proposals, not completed optimizations. Retry thresholds should be based on a material, repeatable complete-request gain after correctness, not the fastest component sample.

| ID | Candidate | Model / evidence needed |
| --- | --- | --- |
| OG-065 | Change source RMS launch thread count | Small rows launching idle threads were identified as a tuning candidate. No changed launch policy or qualified gain was retained; current block size remains 1,024. Must preserve source logical reduction order and validate compiler/dispatch behavior. |
| OG-066 | Remove further KV acceptance allocation/host overhead | Investigated as a small host-cost possibility. Establish an actual allocation/copy in a profile first; OG-045's alleged copy was absent. |
| OG-067 | Larger dense fusion or reusable command region | Profile remaining dispatch/host cost per model, preserve every required BF16 boundary, and measure a new precision-matched ablation. Hybrid graph gains provide no estimate. |
| OG-068 | M5 tensor-assisted prefill | Historical engine research suggests potential. Requires a separate native implementation and numerical/performance qualification across prompt lengths. |
| OG-069 | Gemma MTP/speculative generation | Published assistant-model results were examined only. Requires assistant conversion, target verification, rollback/cache correctness and workload-specific acceptance/speed tests. |
| OG-070 | Broader workload/profile qualification | Longer contexts, prefill/decode-separated timing, sampling settings, other devices, Q8/other quantization groups and other family sizes need independent evidence. Gemma images need UI execution and performance checks if those claims are desired. |

OLMo's remaining opportunity needs new profiling: its latest RMS reuse is already correct but does not move the measured complete request. Gemma's former 2.2% gap is closed for the pinned workload; the next experiment should target a demonstrated bottleneck or expand useful coverage. Repeating the unpromoted sub-1% screens without new evidence would not establish progress.

## Continuing this record

Use [the register/template](../../MODEL_EXPERIMENTS.md) for the next attempt and append IDs from OG-072. Record the original hypothesis, including failed runs and why a candidate was rejected. Bind accepted results to the exact source, binary, artifact, reference and fixtures. Keep model correctness, performance and coverage as separate conclusions. The existing benchmark and validation records remain the source of truth for their named builds.
