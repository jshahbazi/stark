# Inkling 2 Mini decode toward 80 tok/s

These experiments use the original Inkling 2 Mini target and DSpark checkpoint on the 128 GiB Apple M5 Max, with the combined short verifier and batched Q, K and V controllers enabled. The initial build starts at `90fce867` and uses locked Rust 1.98.1. The [integrated controller record](2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-78--integrate-batched-q-k-and-v-controllers) binds the checkpoint hashes, selectors, parity and earlier paired baseline. The target budget is 12.5 ms per useful output.

## 20261122-INKDECODE-79 · Repeat decode on one loaded worker

Run one sealed controller-batch worker. Alternate four greedy requests with three proposals and four seeded temperature-one, top-p-one requests with two proposals. Each window ingests 4,096 tokens with zero reuse, has context 4,352 and emits 256 tokens. Vision and DSpark stay resident. Check the pinned output hash, matching work extent and queued verification on every request. Useful decode covers the 255 intervals after the first output.

| Mode | Median useful tok/s | Range | Rounds across four requests | Median prefill tok/s | Median round time to save for 80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 69.28 | 64.46–72.47 | 398 | 805.77 | 5.03 ms |
| Seeded sampled | 58.17 | 53.10–60.35 | 530 | 804.30 | 9.14 ms |

All eight hashes matched. The worker accounted for 99.84% of observed GPU counter increments across 39 intervals from the first request to the last. Thermal samples were nominal. Pageouts and swapouts were unchanged. Keep this timing as a separate one-worker observation. The earlier QKV paired retry reached 73.18 greedy and 61.91 sampled medians. The request totals and round counts agree across the two sessions.

## 20261122-INKDECODE-80 · Screen smaller NVFP4 workgroups

The saved four-row QKV profile has 13.34 ms in the folded expert region and 2.14 ms in the combined embedding and vocabulary readout region on stock input A. The separate expert-instrumented arm has 7.69 ms across the NVFP4 dot products. Giving the recorded paired workloads all 2.14 ms back per round would model about 77.9 tok/s median greedy, before paying for ranking, certification and exact fallback. Defer the vocabulary shortcut.

Compare the existing four output columns per 128-thread workgroup against two columns per 64-thread workgroup. Keep one SIMD group per output column, the BF16 path, packed coefficient lookup, K16 F32 accumulation and route order. Adjust the X dimension in both the fixed grid and the indirect bucket argument. Restore the same prefix for each paired arm. Use eight alternating plain pairs and four instrumented pairs per width and fixture. Advance to the live worker only for at least 1 ms paired command savings at four rows on both 4K fixtures with full F32 parity.

| Fixture | Three-row paired wall change | Four-row paired wall change | Four-row paired NVFP4 dot change |
| --- | ---: | ---: | ---: |
| Short | +0.288 ms | −0.002 ms | −0.027 ms |
| Stock 4K A | +0.113 ms | +0.146 ms | +0.024 ms |
| Stock 4K B | +0.087 ms | +0.096 ms | +0.027 ms |

Positive values mean the smaller workgroup takes longer. All 504 checked F32 vocabulary rows matched serial. Six paired committed-state checks matched after accepting one input. The first 4K invocation used an incorrect fixture path and yielded no observation. The corrected run passed both 4K fixtures. Archive the smaller workgroup at the native gate. The measured candidate regresses the full command and NVFP4 dots on both 4K fixtures at four rows.

## 20261122-INKDECODE-81 · Replay a privileged sampled proposal reference

Observe the existing worker's exact target logits only as each output is consumed, after the ordinary seeded sampler has chosen the token. Keep the reference calculation in memory. Replay nonoverlapping two-proposal rounds over the same 255 timed outputs. Propose the conditional target mode in each mask. The second target row is valid when the first proposal matches, and a first miss ends the accepted prefix. Apply the ordinary proposal-depth cap at the tail. Charge each request its observed stock cost per round from the earlier QKV paired retry. Reuse those matched prompt costs for the alternate-seed replay.

| Seeds | Stock rounds | Target-mode rounds | Useful outputs per modal round | Modeled median tok/s |
| --- | ---: | ---: | ---: | ---: |
| Pinned | 530 | 461 | 2.21 | 74.00 |
| Alternate, same four prompts | 526 | 450 | 2.27 | 74.34 |

All four pinned full-output hashes matched. The four alternate-seed trajectories each completed the same extents. Pricing the pinned reference at the new one-worker observation gives a 64.35 tok/s median. A separate optimistic replay keeps the modal first proposal, grants a perfect second proposal whenever the first matches and lets hindsight shorten accepted prefixes. It needs 394 rounds on the pinned trajectories and models 85.07 tok/s median at the earlier paired costs. The conditional-mode reference measures headroom for this family of proposals. A future draw-aware policy or a different first proposal can have a different outcome. These rates are replay estimates.

With the later retained MXFP8 vector worker's matched per-request costs, this privileged target-mode replay models **76.03 tok/s median sampled**. Its four workloads require 2.19–2.37 useful outputs per round for 80 tok/s at unchanged cost.

## 20261122-INKDECODE-82 · Fit the existing draft-head bias

Train only DSpark's 2,304-value Markov output bias. Use four archived pinned 4K captures, with both original greedy fixtures sampled at temperature one for this experiment. Draw 192 anchor continuations per fixture. For each held-out prompt, fit on the other three complete trajectories. Use the normalized difference between the frozen unembedding rows for the target mode and the stock proposer as the pairwise training feature. Include a mask-two training pair when the stock first proposal matches the actual sampled first token. Fix the ridge scale and L2 cap before scoring. Round the fitted bias to BF16 and use the existing Metal fold, shortlist and second-mask recurrence. The original target, proposal geometry and draft backbone supply the same work in each arm. Evaluate the candidate with its own first proposal and replay accepted prefixes sequentially.

All four initial F32 target rows matched the pinned captures. The four fits used 242–323 training pairs and changed 463–1,087 BF16 bias values. The folds consumed the same round counts as stock.

| Held-out fixture | Stock rounds | Fitted rounds |
| --- | ---: | ---: |
| Sampled 4K A | 102 | 102 |
| Sampled 4K B | 90 | 90 |
| Greedy 4K A, sampled replay | 84 | 84 |
| Greedy 4K B, sampled replay | 110 | 110 |
| Total | 386 | 386 |

Archive this small bias fit at the held-out replay gate. The fit operates after the base-score shortlist and can only reorder candidates that survive it. The target-mode reference leaves a reason to investigate a broader draft fit or a cheaper verifier. This run qualifies the observed round counts for the tested fit.

## 20261122-INKDECODE-83 · Batch residual mixing across verifier rows

Use Stark `3d5a3fe5`, locked Rust 1.98.1 and the original target and DSpark. The QKV-enabled stock four-row profile attributes 2.65 ms to attention inputs and 1.99 ms to expert inputs. The verifier separately dispatches attention and expert residual mixes for every row. Across 24 cross-pass visits it also launches four gate-message workgroups per row. Batch the attention queries and gate messages together, and run one grid for the expert mix after the fused attention finish. Each workgroup keeps the source 576-thread geometry, F32 normalization, dot reductions and weighted sum. The batching uses the existing per-row buffers and adds no verifier scratch.

Alternate from a restored snapshot with eight plain and four instrumented pairs at each verifier width on two stock 4K fixtures. Require exact serial F32 logits and at least 0.7 ms per-command saving at the steady sampled and greedy widths on both stock inputs before advancing. The plain pairs save the following time.

| Verifier rows | Dispatches removed | Stock 4K A wall saving | Stock 4K B wall saving |
| --- | ---: | ---: | ---: |
| 2 | 143 | 0.730 ms | 0.738 ms |
| 3 | 262 | 1.358 ms | 1.332 ms |
| 4 | 381 | 1.951 ms | 2.068 ms |

All 432 profiled F32 rows matched serial. Separately verify 96 exact continuation outputs and committed state at proposal depths one through three on the short and both stock 4K fixtures. All 45 rejection and ring-wrap controls passed, including reset after an abandoned transaction and rejection of cross-mode snapshots. With flags unset, the connected short and 4K slices passed. The short full target matched pinned F32, and four stock 4K cases matched 16 F32 vectors, 768 BF16 visit residuals and four snapshot replays.

Benchmark on one loaded worker with both pipelines resident. Alternate the selected mix within four paired workloads per mode, using 4,096 input tokens, context 4,352, zero prefix reuse and 256 emitted and sampled tokens. Vision and DSpark stay resident. Both arms use two proposals for seeded temperature-one, top-p-one sampling and three for greedy. All 16 pinned output hashes, accepted and proposed counts and round histograms matched.

| Mode | Existing mix median | Batched mix median | Median paired gain | Faster pairs | Batched range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seeded sampled | 57.77 tok/s | 59.01 tok/s | +3.00% | 4/4 | 58.15–61.48 tok/s |
| Greedy | 68.81 tok/s | 71.15 tok/s | +4.66% | 4/4 | 65.50–78.38 tok/s |

Useful decode excludes the first emitted token. The median paired useful time saving is 0.972 ms per sampled round and 1.664 ms per greedy round. The worker accounted for 99.83% of GPU counter increments across 79 intervals, with nominal thermals and unchanged pageouts and swapouts. Retain `STARK_INKLING_EXPERIMENT_VERIFY_MIX_ROWS=1` under the combined short verifier with fused attention finish. At the observed proposal acceptance, this session needs about 8.42 ms less per sampled round and 4.04 ms less per greedy round for an 80 tok/s median.

The retained source removes the worker arm switch and the extra timing diagnostic. Its sealed native build repeated flags-unset connected short and four-fixture 4K parity, and repeated all 45 enabled rejection and wrap checks. The retained worker restored a 2,048-token prefix across proposal depths, cleared reuse on mode changes and cancellation, preserved the existing request admission limits and matched accepted-proposal stop accounting.

The retained binary's separate one-worker measurement reproduced all eight pinned output hashes and reached **77.39 tok/s median greedy** (75.38–78.95) and **65.18 tok/s median sampled** (62.84–66.81). The worker accounted for 99.85% of GPU counter increments over 36 intervals with nominal thermals. Keep the paired gains and the separate retained throughput estimates scoped to their sessions.

## 20261122-INKDECODE-84 · Vector-load the short verifier MXFP8 products

Keep residual batching and the original checkpoint, draft and sampler. The saved verifier profile places the four grouped MXFP8 visits around 2.4 ms at three rows and 3.0 ms at four. Reuse the existing source MXFP8 vector-load specialization in the short verifier. Four adjacent packed weight bytes and four activation words enter each vector load. Keep the ordered FP32 products, additions, block scaling, SIMD reduction and route order. Both shapes use the same packed banks and workspace.

Restore two stock 4K prefixes and alternate the scalar and vector kernels in eight plain and four instrumented pairs per width. All 432 checked F32 vectors matched serial. Each width won all eight plain pairs on both inputs. The dispatch counts were equal. The four-row expert timestamps attribute 1.51 and 1.07 ms of savings to the MXFP8 products in the two fixtures.

| Verifier rows | Stock 4K A wall saving | Stock 4K B wall saving |
| --- | ---: | ---: |
| 2 | 0.807 ms | 0.824 ms |
| 3 | 1.168 ms | 0.966 ms |
| 4 | 1.450 ms | 1.103 ms |

The candidate reproduced 96 serial outputs and the committed state at each proposal depth on the short and both stock 4K fixtures. All 45 rejection and ring-wrap controls passed. For worker timing, alternate scalar and vector loads on one loaded worker, with residual batching active in both arms. Use the four pinned cold 4K workloads per mode and the same context, sampling, resident artifacts and output extent as the residual-mix study. All 16 output hashes and paired proposal counts, accepted counts and round histograms matched.

| Mode | Scalar loads median | Vector loads median | Median paired gain | Faster pairs | Vector range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seeded sampled | 64.87 tok/s | 65.78 tok/s | +1.55% | 4/4 | 62.78–69.19 tok/s |
| Greedy | 77.41 tok/s | 82.33 tok/s | +5.02% | 4/4 | 74.07–84.79 tok/s |

Three of four greedy workloads cleared 80 tok/s. The median paired useful time saving was 0.449 ms per sampled round and 1.605 ms per greedy round. The worker accounted for 99.85% of observed GPU counter increments over 74 intervals with nominal and fair thermal samples. Retain `STARK_INKLING_EXPERIMENT_VERIFY_MXFP8_VECTOR=1` for the source short verifier. The paired measurements describe this session.

The retained worker removes the diagnostic profiling and arm selector. Its native executable repeated flags-unset connected short and four-fixture 4K parity and all 45 enabled rejection and ring-wrap controls. The worker preserved exact 2,048-token prefix reuse across draft depths, cancellation, mode transitions, draft admission, vision readiness and accepted-proposal stop accounting.

In a separate one-loaded-worker observation with both verifier cuts active, all eight pinned hashes matched. The retained worker reached **84.13 tok/s median greedy** (78.69–87.36) and **68.62 tok/s median sampled** (64.31–69.66). The 80 tok/s threshold was met on 3/4 greedy and 0/4 sampled requests in that observation. The paired worker comparison remains the source for the relative gain.

## 20261122-INKDECODE-85 · Cache normalized residual sources

At `76043606`, the retained sampled run was about 4 ms per round away from 80 tok/s at its measured acceptance. Look for repeated RMS work inside the already-batched residual mixes. After each raw block write, normalize it once into 144 KiB of private verifier scratch. Normalize the two carried vectors in place at the two pass rewinds, after the existing copies have completed. Every query uses the same FP32 normalized values and keeps its source dot, reduction and weighted sum. The changing partial vector still normalizes on demand. Neither scratch nor the temporary carried values join the retained prefix state.

Alternate cache enabled and disabled from the same restored snapshot, with residual batching, QKV batching and MXFP8 vector loads active. Use eight plain and four instrumented pairs per width and fixture. Both arms use the same parameterized mix kernels. All 648 checked F32 vocabulary rows matched serial across the short and two stock 4K fixtures.

| Fixture | Two-row wall saving | Three-row wall saving | Four-row wall saving |
| --- | ---: | ---: | ---: |
| Short | 0.033 ms | 0.092 ms | 0.037 ms |
| Stock 4K A | 0.060 ms | −0.006 ms | 0.050 ms |
| Stock 4K B | 0.051 ms | 0.062 ms | 0.041 ms |

Three-row stage timings attribute about 0.03 ms to attention input prep and 0.02 ms to expert input prep. The complete command gain at the sampled width is too small to advance. Archive the cache at the native gate.

## 20261122-INKDECODE-86 · Score draft candidates with a causal target prior

Use only the already-available target vocabulary row that produced the emitted anchor. In the native INT8 draft ranker, add a signed term of up to one logit unit based on the prior's distance below its maximum, capped at eight score units. Compare reranking the two shortlisted candidates per SIMD group and applying the same term before and after shortlisting. Test the positive and negative terms at both masks, plus positive and negative shortlist terms at mask two only. The unmodified stock ranker supplies the zero arm. The second mask reads the candidate's own first proposal. The original target draw sequence supplies the scoring trajectory.

Use 192 anchors each on four stock 4K prompts and two seeds per prompt. Rotate seven arms per anchor and fit the arm on the six trajectories from the other three prompts. Count nonoverlapping accepted prefixes and verification rounds on both seeds of the held-out prompt. The native worker copies the causal prior into a shared probe buffer outside the timed proposal command. A live implementation would need to bind the resident prior row and preserve its lifetime. The initial patch generator missed the current shortlist declaration and the first Rust build retained the old fixture and arm shapes. The corrected experiment reproduced all eight pinned initial target rows.

| Held-out prompt | Stock rounds | Selected rounds |
| --- | ---: | ---: |
| A | 205 | 204 |
| B | 182 | 179 |
| C | 165 | 166 |
| D | 217 | 216 |
| Total | 769 | 765 |

The best single fixed term on the same eight trajectories uses 763 rounds. Held-out selection saves four rounds in aggregate, about 0.5%, with one held-out prompt regressing. The commonly selected arm adds about 0.005 ms to the median native proposal command. Retain the result as a proposal replay and archive the variant. The exact target and sampler establish the replay reference. The experiment produces no live throughput claim.

## 20261122-INKDECODE-87 · Fuse NVFP4 expert buffer boundaries

At `04fbcca0`, test two changes to the exact half-ingress short verifier with the retained QKV, residual mix and MXFP8 paths selected. First, join the source gate/up epilogue and down-input quantizer in one 256-thread group per sorted route. Preserve the BF16 gate, up and weighted intermediate roundings. Stage the BF16 intermediate within the group and use the source quantizer's order for its row maximum and K16 blocks. Second, scatter the down dot directly to the routed BF16 buffer. Keep the F32 recurrence and apply the same activation maximum, expert maximum and denominator after SIMD reduction.

Measure eight plain and four instrumented comparisons per arm and width, from the same restored state, on the short and two stock 4K captures. The first Metal compile failed with the quantizer's FP pragma placed after the gate loop. Put the quantizer inside its own lexical scope and recompile. All 1,296 checked F32 rows matched serial for the corrected probes.

| Three-row candidate | Dispatches removed | Stock 4K A wall saving | Stock 4K B wall saving |
| --- | ---: | ---: | ---: |
| Gate/up epilogue with down quantization | 44 | −0.036 ms | −0.007 ms |
| Down dot with epilogue | 44 | −0.240 ms | +0.212 ms |
| Both | 88 | −0.198 ms | +0.017 ms |

Positive values indicate less wall time. The combined arm also adds 0.126 ms on the short three-row fixture. Archive both variants at the native gate.

### NVFP4 zero-scale census

Scan all checkpoint tensors used by the original NVFP4 bank's scale buffers. The 20 gate/up banks contain 849,346,560 K16 scales and the 20 down banks contain 424,673,280. None of the 1,274,019,840 scale bytes encode positive or negative zero. There are no all-zero weight-scale slabs to skip across the 47,185,920 output-column wave segments. This census covers immutable weight scales. Dynamic activation sparsity has its own eligibility and cost.

## 20261122-INKDECODE-88 · Recheck sampled draft depth after verifier cuts

Remeasure widths on the sealed retained MXFP8 vector worker, with QKV and residual batching enabled. The verifier cuts saved more wall time at four rows than at three in native measurements. This may change the cost of adding a third proposal. Rotate one, two and three proposals within each of four pinned cold 4K sampled workloads on one loaded worker. Use the same temperature-one, top-p-one draw stream, context 4,352, 256 emitted and sampled outputs, zero input reuse, resident vision and native INT8 draft head. The exact output hashes matched on all 12 requests, and the observed depth and overlap histograms matched each requested schedule.

| Drafts | Median useful tok/s | Range | Rounds across four workloads | Useful outputs per round | Median paired rate change vs two | Faster pairs vs two |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One | 58.87 | 57.00–65.21 | 634 | 1.61 | −7.38% | 0/4 |
| Two | 65.72 | 60.09–69.27 | 530 | 1.92 | Reference | Reference |
| Three | 61.98 | 57.58–64.81 | 493 | 2.07 | −6.75% | 1/4 |

A third proposal lowered the round count on every prompt. On three prompts the increase in round cost outweighed the added acceptance. Keep two at this sampled setting. This test compares the depths within the new session. The worker accounted for 99.85% of GPU counter increments over 58 intervals, with nominal and fair thermal samples and no additional pageouts or swapouts.

## 20261122-INKDECODE-89 · Replay seed-aware draft ranks

At `ace28b28`, replay two seeds on each of four stock 4K captures for 192 anchors per trajectory, using the original checkpoint and native INT8 DSpark shortlist. Sample all four prompts at temperature one. Retain only aggregate evidence. Capture candidates from both Markov masks and evaluate the second mask with the actual first target token as context. Credit its prediction only when the selected first token equals that target. All eight initial rows matched. All 1,536 first-mask winners and all 1,093 second-mask winners reached by the stock first proposal matched the native ranker.

Compare stock rank one with an oracle that selects the actual target when it appears within the top `k`. Count complete nonoverlapping rounds with the usual accepted prefix and tail cap. Price each trajectory at a fixed 28.421 ms per round from the retained sampled run and take the median across trajectories. Actual verification cost varies with the trajectory and the oracle needs the target token in advance.

| Oracle depth per mask | First-mask hits | Two-mask hits | Rounds | Modeled median tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,093 | 660 | 769 | 69.83 |
| 2 | 1,202 | 786 | 710 | 76.77 |
| 4 | 1,293 | 954 | 665 | 84.46 |
| 8 | 1,359 | 1,046 | 635 | 87.79 |
| 16 | 1,386 | 1,127 | 616 | 90.68 |

Fit a policy on the six trajectories from the other three prompts, holding out both seeds together. Each mask and each draft top-two logit gap bucket (<1 or ≥1) chooses stock rank one or one of 20 combinations of temperature and power transform on a copy of the next seeded uniform draw. Normalize the candidate CDF over the full shortlist and fall back to rank one when the draw passes the first 16 candidates. The selected policies used 204, 186, 165 and 217 rounds on the held-out prompt pairs against stock's 205, 182, 165 and 217. The fitted policy increased total rounds from 769 to 772. Archive the calibration at replay. The oracle rates measure shortlist headroom and the experiment produced no live throughput measurement.

## 20261122-INKDECODE-90 · Compare the MLX-Serve Inkling preview

MLX-Serve `v26.11.4-pre-release.1` at `2fe9be834a601c70ef1a597563026340f0e722fe` loads the original checkpoint and the author's 8-bit pack. The launch benchmark reports a best single-stream decode of 116 tok/s on M4 Max and 200 tok/s on M5 Ultra, using the 8-bit pack and bundled drafter across llmprobe 0.6.18 scenarios with greedy sampling and 192 output tokens. The model card lists a separate M4 Max observation of 84 tok/s serial and up to 96 tok/s drafted. The pack preserves the NVFP4 and MXFP8 experts and quantizes the attention projections, embedding and vocabulary head to group-64 affine 8-bit. [Release](https://github.com/ddalcu/mlx-serve/releases/tag/v26.11.4-pre-release.1) · [Launch benchmark](https://x.com/ddalcu/status/2124377137821925651) · [Model card](https://huggingface.co/ddalcu/Inkling-2-Mini-MLX-8bit)

The pinned plugin proposes up to seven tokens, caps the width by the product of draft confidences against its observed verification cost and commits the accepted prefix from an open forward. The verifier reads packed expert banks directly at two to eight rows and shares KV reads across rows with a split relative-attention path at four to eight. Stark's prior depth gate found an added 1.10–1.26 ms when waiting for draft confidence. Test width and fence costs together before importing the policy. The affine 8-bit path needs its own artifact and validation. [Plugin design](https://github.com/ddalcu/mlx-serve/blob/2fe9be834a601c70ef1a597563026340f0e722fe/src/plugins/inkling2/AGENTS.md#L35-L79) · [Open verification and commit](https://github.com/ddalcu/mlx-serve/blob/2fe9be834a601c70ef1a597563026340f0e722fe/src/plugins/inkling2/plugin.zig#L370-L452) · [Stark confidence-depth screen](2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-70--price-draft-confidence-depth-selection-after-queued-int8-proposals)

Run the macOS arm64 release, verified against its published archive digest, on the local 128 GiB M5 Max with the original checkpoint. Enable the bundled draft head and leave vision enabled. Use one server on loopback and a discarded warmup. Rotate draft-off and draft-on requests over two greedy and two sampled captures from the retained Stark comparison. Send their exact 4,096 prompt IDs, set context to 4,352 and prefill chunks to 2,048, disable prompt lookup, and confirm zero reused tokens. Each request returned 256 tokens. Sampled requests use temperature and top-p one, no top-k or min-p, and the pinned input seed passed to MLX-Serve's sampler. Calculate useful decode rate as 255 divided by the server's native decode phase. Stark values are the corresponding retained-worker observations from the earlier session.

| Workload | MLX-Serve draft off | MLX-Serve draft requested | Stark retained | MLX-Serve paired completion text match |
| --- | ---: | ---: | ---: | --- |
| Greedy A | 69.03 | 73.23 | 87.36 | Yes |
| Greedy B | 67.71 | 66.14 | 78.69 | No |
| Sampled A | 68.10 | 78.27 | 68.10 | No |
| Sampled B | 65.83 | 69.08 | 64.31 | No |

Rates are tok/s. Draft-requested medians are 69.69 greedy and 73.68 sampled. The paired text hashes differ on one greedy and both sampled workloads. The API receipt compares decoded completion text and does not expose an exact token parity or accepted-depth trace. This timing screen used the original weights and leaves the 8-bit launch benchmark unreproduced. It supports measuring verifier cost at wider short widths on Stark while preserving the pinned output checks.

## 20261122-INKDECODE-91 · Price wider verification and local KV sharing

At `1345327f`, use the original checkpoint and bundled INT8 DSpark proposer on the 128 GiB M5 Max. The current rollback verifier accepts up to four rows. Its local attention shares one head's cache reads across two queries, while its global attention shares three Q heads on one token. Test the additional local geometries at three and four verifier rows before changing the rollback capacity.

Each geometry retains the original query's key order, physical-to-logical shard mapping, independent FP32 accumulators and use of the pre-write cache for displaced keys. Hold the other retained optimizations fixed. Rotate four arms within each input, with eight plain and four instrumented repeats at each width. All 1,008 checked F32 vocabulary rows matched serial. The 180 controls verified accepted state and the next continuation across the short prefix, two stock 4K prefixes and their local-ring crossings. The short prefix bypasses shared local attention.

| Local sharing per group | Three-row 4K paired wall saving | Four-row 4K paired wall saving |
| --- | ---: | ---: |
| One head, up to four rows | −0.029 to −0.011 ms | −0.213 to −0.179 ms |
| Three heads, one row | +0.154 to +0.218 ms | +0.278 to +0.326 ms |
| Three heads, up to two rows | +0.076 to +0.099 ms | +0.174 to +0.208 ms |

Positive values mean less wall time against one head with up to two rows. The strongest arm reduces the measured relative and cache attention stage by 0.170–0.180 ms at three rows and 0.304–0.310 ms at four. Keep this experiment as exact component evidence. These savings alone leave the 80 tok/s sampled round budget open.

### Wider target cost

Use the existing direct grouped-expert verifier diagnostic at the same two restored 4K prefixes. The first build used a previous diagnostic's build cache after an invocation with Cargo missing from PATH. The follow-up built in a dedicated target directory and added widths five and seven and separate stage timing. All grouped parallel-router arms matched the complete F32 readouts and committed target state in both runs. This graph uses the wide diagnostic's direct append. Its timing excludes draft proposals, undo, sampling and the retained four-row transaction's shared KV and fused finishing paths.

| Verified rows | Grouped parallel-router wall time at 4K |
| --- | ---: |
| 3 | 27.83–28.15 ms |
| 4 | 31.30–32.61 ms |
| 5 | 40.64–42.59 ms |
| 6 | 46.12–49.38 ms |
| 7 | 50.85–55.34 ms |
| 8 | 55.12–60.06 ms |

Five rows add 9.33–9.99 ms over four in this graph. Separately timestamped commands account for 5.54–6.53 ms of the increase in the NVFP4 expert region and 1.22–1.41 ms in relative and cache attention. Above four rows the graph falls back from the compact route launch and short native router. The first run measured 45.51–48.65 ms at six rows and 54.63–58.17 ms at eight. Widening the optimized transaction would require replacing fixed four-row bindings and preserving any accepted convolution history overwritten by a rejected suffix.

### Exact acceptance through seven proposals

Replay the native INT8 draft head at 192 consecutive anchors on each trajectory. Use the exact target's seeded temperature-one draws on four 4K prefixes and its greedy continuation on two of those prefixes. Each proposal conditions on its own preceding proposal. All six initial F32 rows matched the pinned traces. Rotate proposal depths one through seven at every sixteenth anchor. All 432 shorter proposal sequences matched the corresponding prefix of the seven-proposal sequence.

Replay each depth as nonoverlapping rounds with the usual tail cap. The optimistic sampled rate charges every width the incumbent depth-two round cost of 28.421 ms. The 80 tok/s budget is the median per-trajectory cost allowed by the observed round count. Neither estimate includes any increased cost of widening.

| Drafts | Sampled rounds, 768 outputs | Greedy rounds, 384 outputs | Optimistic sampled median tok/s | Sampled round budget for 80 tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 457 | 212 | 59.55 | 21.16 ms |
| 2 | 386 | 171 | 70.65 | 25.10 ms |
| 3 | 354 | 165 | 79.34 | 28.19 ms |
| 4 | 347 | 164 | 80.58 | 28.63 ms |
| 5 | 345 | 164 | 81.56 | 28.98 ms |
| 6 | 341 | 164 | 82.16 | 29.19 ms |
| 7 | 338 | 164 | 83.39 | 29.63 ms |

The deeper draft path yields real acceptance headroom. At four proposals it allows only about 0.21 ms above the incumbent round cost, and the draft command alone adds 0.19–0.20 ms at the paired median across the four sampled fixtures. At seven proposals the extra round allowance is 1.21 ms. The native target timings leave no credible budget for the current wide design. Defer expanding the rollback verifier. This is a sizing and acceptance experiment with original target outputs and no live worker rate for the new geometry.

## 20261122-INKDECODE-92 · Reconstruct packed BF16 bytes across verifier rows

Start from `c4e5bf00` and the retained verifier configuration. The dense attention projections use about 5.7 ms at three rows. The earlier [lossless singleton screen](2026-11-21-inkling-mini-decode-60.md#20261121-inkdecode-15--lossless-bf16-attention-weights-on-metal) spent extra time reconstructing sign and mantissa. Preserve each BF16 word's low byte verbatim. In a 64-word chunk, encode the high byte as a sign bit and three high-exponent delta bits. The subgroup reuses each reconstructed four-word vector for all live rows. Keep the source BF16 casts, ordered FP32 dot products and lane reduction.

Across 176 original attention banks, 10,042,333 of 10,093,824 chunks (99.49%) fit an arbitrary base. An eight-aligned base admits 9,305,048 chunks (92.19%) and permits bitwise reconstruction. Ineligible chunks read the original BF16 data. At payload level, the two formats remove approximately 24.1% and 22.3% of bytes, including base bytes and raw fallback. The diagnostic retains the original banks alongside both packed planes. Check every packed word on CPU, then compare arbitrary-base shifts, arbitrary-base byte-pair lookup and eight-aligned shifts on the physical GPU.

Use the short prefix and both stock 4K prefixes, with eight rotated plain and four timestamped comparisons per width. All 1,296 full F32 readouts matched serial at widths two through four. All 36 checked single-row commits matched the serial target and draft state. The first Rust build could not access the Metal registrar from the packer's module. The timed build registers from the verifier diagnostic.

| Candidate | Three-row 4K paired wall increase | Four-row 4K paired wall increase | Three-row dense stage increase |
| --- | ---: | ---: | ---: |
| Arbitrary base and shifts | +2.016 to +2.199 ms | +2.013 to +2.142 ms | +2.006 to +2.107 ms |
| Arbitrary base and byte-pair lookup | +1.877 to +2.030 ms | +1.883 to +2.015 ms | +1.821 to +1.906 ms |
| Eight-aligned base and shifts | +1.970 to +2.133 ms | +2.020 to +2.070 ms | +1.883 to +2.010 ms |

Archive the packed layout. Sharing reconstruction across all three rows and using a byte-pair lookup still raises the dense stage time. The low-byte layout preserves exactly the checkpoint weights and the numerical reference in this component screen.

## 20261122-INKDECODE-93 · Run independent attention projections concurrently

Keep the original BF16 matrices, per-row arithmetic and dispatches. The Q, K, V, relative, gate, sink and dynamic projections read completed inputs and write disjoint destinations within one visit. Place these projections in a concurrent Metal compute encoder on the same target command buffer. The preceding and subsequent encoders keep their normal ordering. Compare the incumbent command, a separate serial encoder for each projection group, and a separate concurrent encoder. The separate serial arm isolates the effect of changing encoder boundaries. Timestamped arms select the corresponding dispatch mode inside the existing attention stage encoder.

Rotate eight plain and four timestamped repeats per width on the short and two stock 4K prefixes. All 972 checked F32 readouts matched serial at widths two through four. All 27 checked accepted states matched serial target and draft state. The concurrent arm won seven of eight plain comparisons at both active widths on each stock fixture.

| Verifier rows | Stock 4K paired wall saving | Stock 4K attention dense stage saving |
| --- | ---: | ---: |
| 2 | 0.770–0.791 ms | 0.738–0.813 ms |
| 3 | 0.730–0.828 ms | 0.803–0.815 ms |
| 4 | 0.728–0.892 ms | 0.876–0.930 ms |

The three-row serial encoder control saved 0.023–0.066 ms of wall time. The concurrent arm added 0.05–0.08 ms of CPU encoding at three rows. Advance the concurrent arm to the exact worker comparison.

Run four alternating pairs per mode on one loaded worker with the original target, vision and INT8 DSpark resident. Use cold 4,096-token prompts, context 4,352, 256 emitted and sampled tokens, two proposals for seeded sampling and three for greedy. Keep the pinned outputs, paired proposal and acceptance counters, and round histograms equal. Advance on a sampled median paired gain of at least 2%, at least three sampled wins, a nonnegative greedy median and no individual regression exceeding 5%.

The worker build passed connected short and 4K parity with flags unset, including 16 stock F32 vectors, 768 BF16 residual blocks and four exact snapshot replays. With concurrent projections active, all 45 rejection and ring-wrap controls passed. The first scoring wrapper stopped after proof because it could not open the timing logs. A second invocation lacked the converter environment. Both stopped before timing generation. The scored run uses the existing locked converter environment and the sealed binary.

| Mode | Control median | Concurrent median | Median paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Seeded sampled | 65.48 tok/s | 63.78 tok/s | +0.77% | 2/4 |
| Greedy | 79.76 tok/s | 80.77 tok/s | +2.20% | 4/4 |

All 16 output hashes and paired work counters matched. The median paired useful saving was 0.23 ms per sampled round and 0.70 ms per greedy round. One sampled request regressed 7.69%. The worker accounted for 99.86% of observed GPU counter increments over 49 intervals with nominal thermal readings and unchanged pageouts and swapouts. Disk pressure required reclaiming the experiment's generated build cache during the session. Measure the repeat without a build-cache cleanup.

The first sampled session missed this gate. Repeat all four sampled cells on the same sealed binary, with the order of workloads and paired arms reversed in a fresh loaded worker. The repeat requires the original sampled gate, at least six of eight combined sampled wins and a pooled median paired gain of at least 2%. Keep the first session and repeat separately visible.

The repeat matched all eight pinned hashes and relative work counters. The control median was **68.53 tok/s** and the concurrent median was **67.94 tok/s**. The candidate won one of four pairs and the median paired change was **−2.97%**. Across both sessions, sampled won three of eight with a median paired change of −1.44%. On the repeat, the queued verifier interval grew by 0.78 ms per round at the paired median. The worker accounted for 99.81% of observed GPU counter increments over 36 intervals, with nominal and fair thermal readings and unchanged pageouts and swapouts.

Archive the separate concurrent encoder at the worker gate. The native snapshot commands benefit from concurrency, and the live sampled schedule regresses in the repeat with the draft and target queued together. The queued measurement includes CPU encoding, competing work and the GPU completion wait. The experiment does not isolate which of these caused the difference.

## 20261122-INKDECODE-94 · Compare concurrent schedules in the queued verifier

At `247342b6`, retain the serial schedule and the earlier separate concurrent projection encoder as controls. The new arm uses one concurrent encoder for the target body. It issues buffer barriers before dependent direct or indirect dispatches, and one barrier on entry to the independent projection group. The next dependent dispatch orders the group's writes. Leave the target preparation, DSpark submission and commit on their existing encoders.

Use eight restored states per depth and both submission paths. The standalone target receives the exact proposal IDs from a serial-encoder queued control. The queued target reads the real INT8 draft output. The first harness used the synchronous BF16 proposer as the one-proposal reference and stopped on the first 4K fixture after its proposals differed. The corrected run uses the queued reference for all depths. Every queued proposal sequence, all 1,644 F32 vocabulary rows and all 540 committed target and draft states matched the corresponding references. The count includes short and two stock 4K fixtures and rejection through the local ring crossings.

| Verifier rows | Separate encoder standalone saving | Single encoder standalone saving | Separate encoder queued saving | Single encoder queued saving |
| --- | ---: | ---: | ---: | ---: |
| 2 | +0.106 to +1.500 ms | −1.815 to −0.202 ms | +0.572 to +0.715 ms | +0.143 to +1.014 ms |
| 3 | +1.112 to +1.877 ms | −0.433 to −0.331 ms | +0.678 to +1.258 ms | −0.478 to +0.218 ms |
| 4 | +1.196 to +2.174 ms | −0.129 to +0.138 ms | +0.719 to +0.878 ms | +0.054 to +0.226 ms |

Ranges span the paired medians of the two steady 4K inputs. Positive values save time against the serial encoder. Queued timings cover submission of the draft through target completion. The separate encoder saves native time at the sampled width, with five wins out of eight on each stock input. The single encoder wins four of eight on each stock input at that width and fails to produce a consistent saving. Archive the single encoder at the native gate. The previous separate-encoder worker decision stands on its end-to-end evidence.

## 20261122-INKDECODE-95 · Screen an exact vocabulary readout with INT8 bounds

Reuse the resident DSpark INT8 vocabulary weights on the target's exact BF16 hidden row. Apply the same source logit cap. Give each coarse logit a screening interval from the BF16-to-INT8 residual norm and the input's norm. Add a conservative allowance for tensor rounding and the cap. Select the largest upper bounds and use the original BF16 scores at those IDs. An omitted upper bound limits which exact scores can form the initial sorted prefix. Bound the total omitted F64 probability mass below and above, and accept a draw only when both ends use the same exact prefix token. An unresolved draw would take the full original head. This experiment checks the proposed interval empirically against the full head.

Use four stock 4K captures at 128 chronological temperature-one draws per capture. All pinned initial F32 readouts matched. There were zero envelope violations across 102,429,696 score comparisons. The largest individual capped INT8 error was 0.9334. The references replay each prompt with the ordinary seeded target sampler.

| Candidate scores | Exact maximum certified | Seeded draw interval resolved |
| --- | ---: | ---: |
| 64 | 511/512 | 475/512 |
| 256 | 512/512 | 493/512 |
| 1,024 | 512/512 | 503/512 |

Recapture the distinct original 32-token tiles for the 256 candidates every 16 anchors. Launch one gathered-tile grid with the source 16 × 32 BF16 product and the ordinary cap. All 152,800 checked scores match the full head bit-for-bit. Median tile counts range from 122 to 171 per trace. Isolated phase medians across the traces are 2.187–2.443 ms for the full BF16 readout, 1.024–1.051 ms for the coarse INT8 head, and 0.077–0.176 ms for exact tiles. Charging a complete BF16 readout on every unresolved draw leaves an optimistic 0.966–1.078 ms GPU saving before orchestration and the 1.363–1.579 ms CPU bounds screen. These timings use one live hidden row and the source's padded tensor widths.

### Metal upper mass

Repeat the four trajectories with a Metal pass that bounds omitted probability mass from above and uses zero as the lower bound for omitted mass. Round each upper logit into the next 1/64-logit bin. Use an upward-rounded exponential table, reduce 32 values per SIMD group, and add an allowance before combining group totals on the host. The CPU selects 256 upper candidates, subtracts the selected candidates' contributions from the upper mass, then uses those candidates' exact scores. The 256-candidate interval resolves **489 of 512 draws** with zero observed upper-envelope violations and the same bit-exact selected-tile controls. The CPU procedure still uses the full head as the reference source for the selected logits. The exact tile kernel supplies a separate cross-check and cost measurement.

Metal uses **0.021–0.032 ms** for the upper mass pass, and the remaining host work takes **0.255–0.304 ms**, including host readback, envelope validation, candidate selection and the interval check. A model using the same-session phase medians and a full head at each observed miss leaves **0.475–0.637 ms** per readout before submission overhead. The extra standalone command's measured wall cost is **0.565–0.695 ms**. Charging that boundary moves the modeled savings to **−0.069 through +0.071 ms** across the four traces. Keep the device bound and tile kernel as feasibility evidence. Any live experiment needs the bound and coarse head in the same command, candidate unions across all live verifier rows, an exact fallback, and a source arithmetic bound suitable for use beyond the observed trajectories.

## 20261122-INKDECODE-96 · Time sparse vocabulary readout inside queued decoding

At `92ea0686`, use the original target and resident DSpark INT8 vocabulary bank. Replace the full BF16 head at the end of the queued three- or four-row verifier command with the coarse INT8 head, hidden norms and upper-mass reduction. Compact every upper logit within nine units of the highest upper logit, retaining up to 256 IDs per row. An overflow takes the top 256 on the host. Select the union of the original 32-token tiles across all live rows and launch one BF16 rescore with the source product and cap. Prove consumed greedy or seeded temperature-one draws using those scores, and submit one full BF16 readout if a consumed draw needs it. The diagnostic still uses the empirical arithmetic allowance from attempt 95.

Run two pairs per mode on one loaded model per fixture, with the arm order reversed across the two trials. Each arm produces 128 tokens from the same snapshot, using two proposals for sampled and three for greedy. Change the sampled seed between trials. Time the 127 useful output intervals including the queued draft and target, candidate selection, exact rescoring, consumed draws, fallback and commit. The four 4K seed readouts matched pinned F32. All 20 short and 4K pairs matched every emitted token, proposal and acceptance total, round count and final committed target and draft state.

| Fixture | Mode | Full head median | Sparse head median | Median paired gain | Faster pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| Short | Sampled | 60.61 tok/s | 64.30 tok/s | +6.08% | 2/2 |
| Short | Greedy | 103.53 tok/s | 110.88 tok/s | +7.15% | 2/2 |
| Four stock 4K traces | Sampled | 59.83 tok/s | **65.54 tok/s** | **+8.05%** | 7/8 |
| Four stock 4K traces | Greedy | 58.67 tok/s | **61.91 tok/s** | **+5.01%** | 8/8 |

The sampled 4K arm certified 922 useful outputs in sparse readout and fell back on 79 of 517 sparse rounds. Greedy certified 1,004 and used zero fallbacks across 480 sparse rounds. The median per-run rescore took 0.30 ms per round in both modes. The paired median queued interval saved 1.08 ms sampled and 1.26 ms greedy. Complete timed rounds saved 2.32 and 1.72 ms at the paired medians. One sampled pair regressed 1.27%. Sampled sparse rates across all eight paired reads ranged from 49.33 to 82.05 tok/s.

Advance the probe to an arithmetic and worker gate. The experiment compares native decode from private snapshots. An exact runtime path still needs outward-safe bounds for the source tensor and cap, selected-score and forced-fallback checks at all live widths, and a same-loaded worker comparison with pinned greedy and seeded output hashes. No claim of live worker throughput follows from these measurements.

## Evidence

The original target and DSpark artifact digests are recorded with the [integrated QKV run](2026-11-21-inkling-mini-decode-60.md#evidence-identities). The worker and native screen retain aggregate-only private receipts. No prompts, token IDs, logits, targets or fitted tensors are published.

| Evidence | SHA-256 |
| --- | --- |
| Sealed integrated worker | `638741dc417d72591a96b3fe5f4bccac64b345950319c09bfcded6d11100b485` |
| One-worker harness | `d950570aad9ffa2ae0f0a5e46f4e608096836ea99535ded1ddcf59a47220d7d9` |
| One-worker receipt | `8bb65ca242a3db7388f64bbbf2d7a2425918847e5380979b98ea1f47475bd261` |
| One-worker host observations | `da15251353b67f433c9956d4c9a67c7b01fe62afd6233792ce8b04a6961dbf85` |
| NVFP4 group probe source patch | `783b0e29a9d763f008aa7c59143ec9a95944acc10b545be2f96f81334dad3f7a` |
| NVFP4 group probe binary | `3b2333906678b1ead912edbb1f090c710a1a330e8d018ede5b0a751357bd0ea2` |
| NVFP4 group probe, short | `971aa793ea5f7a96db2863b11d6fb5ac78d96d5eb54de19ed393b627f92456f7` |
| NVFP4 group probe, stock A | `bbff9708a86baaf3331fee6dad815d1f93e2026ed51b40aeaa3f1a58123b612b` |
| NVFP4 group probe, stock B | `8aad87a1989d786a065e6b3d4216aa65e3030bb8f044208cc585a0615a237bf2` |
| Target-mode source patch | `e248b1c05ff9a16f4f6e5f13bad814971287b06b139f584247f9ac6173c403ca` |
| Target-mode worker | `9dd015d2a1660e5cff34ea0b5aced3b7d17b79920fe60abb3063ae26733abc41` |
| Target-mode harness | `0bf8ef50a4042f7c21a7d0d0844550c99d25439419037de09ee000aa81704ceb` |
| Target-mode replay receipt | `91870d7393ca1f18bef57dfd43cbd288c44ee01dca2838f55a2b314008ac568a` |
| Draft-head fit source patch | `3606c0ef03d0524c4da037b9a4f8bab1f7e501b6ded27ac9ac3905d596cec0b1` |
| Draft-head fit native binary | `4406b126ba92da66c9e96b9fa51345e101eb2cececf7adf4e97a5a94bec6ad4f` |
| Draft-head fit harness | `62e1018dcb883135dc93973771761f5eef06906da2e1564560011f55dfe1c99a` |
| Draft-head fit receipt | `3ef73c1e25e0cb5b355ea014794a41048284cc63587ae8191a582476dd29c8ba` |
| Residual mix paired worker source patch | `a14338d5f16c17c11987cd433ed788ca149583b51f345eba3ffb9499a604f5d8` |
| Residual mix native profile source patch | `fe217c13ffe7152c14f1e27caaec4bddfc6968397711ba658e7584e9cf545556` |
| Residual mix paired worker | `ca0cc73012081bcf9c8f3f271c81eaa115de9f019bfbd467aac188fad2025ee1` |
| Residual mix native profile executable | `53fb03931ecf90f9ba9c85f475d12564d10c7a47ec15777d9e51a28677c130d8` |
| Residual mix paired benchmark harness | `ccf080ab744e6e56ae0943a4524f6a207c61e305cbbca12de5c26bc68abe592b` |
| Residual mix paired benchmark receipt | `819e1c07300756c73f975e35303cbd0235b4a0b2cfeb111149ff2075afc12279` |
| Residual mix paired host observations | `223327eeab5abe7db54d57b4cf6c99a48f7045b76131cd3d6bb19c6eb175ea38` |
| Residual mix native profile, stock A | `6c8d621c931a6141317398ea61c06843daf1f3cbf6ed1048913b21db42f5292d` |
| Residual mix native profile, stock B | `a4400e402a66dafa726625de9d330d74035f690aeeb3f8e7d752d5f7bfc8b895` |
| Residual mix retained source patch | `1b3b55514d03caeb690d7eb7955fa234c4d46be6a82181f2c26e8bf7cbb43c53` |
| Residual mix retained worker | `220271d9a2c84f5941aba1bf5f95ebfb55dc21ef426daa34f60f4d83a335e050` |
| Residual mix retained native executable | `e7f6c3fab30d24b8f62ad4e98fad35630748984449997031d326c6b6a04c7865` |
| Residual mix retained worker lifecycle | `d1b51cd09652bd043eb12d6546e89535af29ed286b5f587400a88200975afc6b` |
| Residual mix retained worker benchmark harness | `a292a5840c7a204e9a6b84bc08e9b68c4e62450a7379f0b0d9c76d28217817d2` |
| Residual mix retained worker benchmark receipt | `b72b9cd4d3983714999f403b87153d683b08744e79d0b4b6da59547b8772abf4` |
| Residual mix retained host observations | `b47e3f8d19ce2fece22b7a19e6ff6c7eb933bb595ed772e4d8208cd7b6071ff2` |
| Residual mix retained proof, short | `2713ca77ee793c470bd0ad47e005b517cf8a9be306fd18921cbd18c8b407070c` |
| Residual mix retained proof, stock A | `e16b1fd568a50ac8753eb27a132f5bd3bd172d4e8ec76777c9ca8f315558a4db` |
| Residual mix retained proof, stock B | `7a5a70b041a0238a104bb08921ffef55d0465812d11beb856b9bd850d64c508a` |
| MXFP8 vector profile patch against 3d5a3fe5 | `76a07f72dd54f00c9b2090e0e2dc2cea901a08bbca012eddfb2f5ad72cc4cac0` |
| MXFP8 vector profile executable | `f6357b8bb5351d8ab5200a3b1ad8968c4542dd981fdbc013914c94ea2a5cbc07` |
| MXFP8 vector profile, stock A | `98e69b65818a213e17a1d421bb2c7849942191f846eef28832ce5ca13a621a28` |
| MXFP8 vector profile, stock B | `1785fc065f5b714caba5f668d102db2125e7fc5f3434d3fd150db8a9ddc2889e` |
| MXFP8 vector paired worker patch against 3d5a3fe5 | `e7b2a6bb8253c459be8b3f75fa9f53a13c203eec3277935be9d9f2167cc9843f` |
| MXFP8 vector paired worker | `3319ba8e0bbc18407d54e8f320b29bf0ab63b05002d85c698c20edd2da54b6ec` |
| MXFP8 vector paired worker harness | `5aa3fff73b0962993b0839f85fc5d00312210a9c7bdf3e9b55d5944a03d230d7` |
| MXFP8 vector paired worker receipt | `4dd826755961337b48051bc753e4df13d95746e00f90df13800adfb15e91565d` |
| MXFP8 vector paired host observations | `b28c3999e1c520c2b373b0281d809a09c8ed610610659a043721263988aeeffa` |
| MXFP8 vector retained patch against 821c55be | `63c92804f305803e85e8ee8a7f3aea87134bf448fba658d11c7651160c0ce70a` |
| MXFP8 vector retained worker | `42b4a068c3c2ee153f8e80dd284a073c36551c0952030f2a1e83153520627602` |
| MXFP8 vector retained native executable | `82f86ff403f959855dcebf18336b72a515f7060493205fb38f2c7ac5340dd91d` |
| MXFP8 vector retained worker benchmark harness | `d622d7ae6218d0818f99b75774a33b9b7c3e1ed716aa4d0938b98d8c7251fe97` |
| MXFP8 vector retained worker benchmark receipt | `e19639164600a2d0a2961f3d18a49e921b9587e8485bdf0d20e3c63972f7a901` |
| MXFP8 vector retained host observations | `abb5d47844f49016603b37dc6769725f6b923f1e0a68372896f5b2cfb5a0b09d` |
| MXFP8 vector retained worker lifecycle | `d1b51cd09652bd043eb12d6546e89535af29ed286b5f587400a88200975afc6b` |
| MXFP8 vector retained proof, short | `9d4a4e67ab1fc350ef402e11c8d879ae526a15c7bc7f360de122ac1027042b62` |
| MXFP8 vector retained proof, stock A | `fa2abb938322ff0299f69e7d0f7aa00c7c13b8e1d09601e4046cb79a99590ceb` |
| MXFP8 vector retained proof, stock B | `0e3dd7c36224f9bdb08da10c33fe2a24973ecdce34b012e671b06891c4c55e77` |
| Residual source cache diagnostic patch | `431d7395093503efdd58cc84fcf9a5553b75a9c8074472a49ebe7627c770e36c` |
| Residual source cache native executable | `09548f1c5b9b210ebdec875dd19eb70f01512984f23b6643126075338f364af6` |
| Residual source cache short receipt | `12b6ee440485ce68df02e58f38af8dce2975329897c636c96ab898354a7d4ab0` |
| Residual source cache stock 4K A receipt | `d1df9b4ea54b1dee6bea60d8bf5ae2129f33bb1bc15b6f9feb808c51b919a9ae` |
| Residual source cache stock 4K B receipt | `63fd439f183c6994d5ad965c979d72ddac7d45d758b57f5b21cea5da70f31f2a` |
| Causal draft prior diagnostic patch | `8e3687bb34a2233e5245778b81410d8328c5a43afea77a82e2c72a7446f3a585` |
| Causal draft prior native executable | `487884e44084dd3d8f84d9a0590eaaf01d017b1c9177b0eb2ad62817b53ff5e5` |
| Causal draft prior replay harness | `3626d04b8e3d15f52e514b2e9f52498ccf0cf864016e6dc7e43987ca9ee06a60` |
| Causal draft prior replay receipt | `15d991f30263f1b823c1a79dd03a9a9288dcdfb9e9f24ea34ba0fe423d833d67` |
| NVFP4 expert boundary two-part fusion patch | `629ce09f0b8332f3f0ae35d7b9248c4d514abb999db2d1e937635d65c4b403bc` |
| NVFP4 expert boundary native executable | `abc0d41084016a35218e2cf330e49a2391e2849e270b8da9ccd01f96bbbf6be0` |
| NVFP4 expert boundary short receipt | `afcfbb5f390fbfbf9b01af13a8a237a70cfcde0adf179bd6ee29a14cb67f0c75` |
| NVFP4 expert boundary stock 4K A receipt | `4d811e115af8b5c390c61950253b5384bcf74605d41eb78f229f7749d990051d` |
| NVFP4 expert boundary stock 4K B receipt | `52851ae0b334d570d5ee817e7cfe6fc0d23973b4a2b5d71bd9159c4d4f31b7e6` |
| NVFP4 zero-scale census script | `ec68f2d7b75632415f5cc3a07718c937215f1ff7c53b64ed8ad36e985ef7cddd` |
| NVFP4 zero-scale census receipt | `7262caa91fe0e6528d9722f995f374b3894d5c50185b26d8e5b5671c46b7b7ad` |
| Retained sampled depth worker harness | `69b374deba4722b4359de4b9b3ec838bbb8abe20481415842ec1ec272c96a7f1` |
| Retained sampled depth worker receipt | `5a24492801d59c191df516e259e3605f77132f8f56ddfdd5cd24c1615a6d802f` |
| Retained sampled depth host observations | `52dfa1a706fc0e4d64f43b9a5b1b4706be437447df48f6a982e04bf498d65f87` |
| Seed-aware rank replay patch | `437dbc2b8611d9ecf75cf458e4b9ee1e19d787125a986c826167d44bc7ef3baf` |
| Seed-aware rank replay executable | `8de565c03d5d622984abcca1ebe066a7a47ec4715d9e711c7fd2bee9cfef9f2f` |
| Seed-aware rank replay harness | `15670ab144a0d5026faf7a72e61bb90727ae0048fed586328b7640027d05ed9f` |
| Seed-aware rank replay receipt | `db840193bd825752d9fed754d336ff452ecb0fdf202d88bcc802a7b945a51a82` |
| MLX-Serve macOS arm64 release archive | `510de30e3db01786098e7022dddb84158462e2f2333b08dde1a63e0e3ccbc65e` |
| MLX-Serve executable | `05983e1278d6253d3648ddaa2514ec2a2f73ff3f617d41412d18334759991392` |
| MLX-Serve original-checkpoint harness | `8820cd5dee67c14c3696ef271defbd57883cfd6a15688bb417853a52d8cc7f37` |
| MLX-Serve original-checkpoint receipt | `7a8f803677591cb9857f8173d5a9f5d52f203deda0b42a1d75a083ef9bb6454f` |
| Local attention and wide draft probe patch | `c4878c88fe84b6274f84d0c2fdb2f137359263600b0bef1ffc1df2088c5c1fb1` |
| Local attention and wide draft probe executable | `489f80bfaa65a856a20593990a5e6753596515f3b76d38cde801a455e66207b9` |
| Local attention and wide draft probe harness | `93019034b456cff828c35760c0bc811b03042f1010a73d6f67018932a70c22f5` |
| Wider target cost follow-up patch | `5c4ca8d7e7a1c933edcac4b0f9cb6df4c3783982a83ef60a648adb91ddb6c4f1` |
| Wider target cost follow-up executable | `530f84bc5f4fdd77a939c71091ff49d96911415551b66c65af3d27c50d7f3394` |
| Wider target cost follow-up harness | `ce01f28c83fcbe6dacfdefcaed8be958aad1b61fbc57c4f4018b27c05e752164` |
| Wider verifier and draft aggregate receipt | `7ef289b5237063a0a06044ca0e6795ac21c135ebca3abd36482e59b3fe43fd8f` |
| Shared BF16 high-byte packing patch | `0ca295bd6d593165753ffcaeacad5c2558ab73c78116977f5370d923e54e500c` |
| Shared BF16 high-byte packing executable | `a6a985f79036220efb22dbf1039e0f6b44b63548c7cb2b0c637416b993f488a0` |
| Shared BF16 high-byte packing harness | `5bf67d2f3fe9bc64dbc0284fd64ee4096e00d41f4eb61805003b062ecacf5a5b` |
| Shared BF16 high-byte packing aggregate receipt | `b11eb1f19136c4da000319ec26ca6b683738aee33bea8409d987e827e28b344f` |
| Concurrent dense native probe patch | `03b5b3eca6220f02007f10267b0630db7d072111566405eff02272eee863b638` |
| Concurrent dense native probe executable | `2a780229a2ff9661e8be675e87beb09a80c0058efceff078df0fa25eb950e2cd` |
| Concurrent dense native probe harness | `31bc52deeee5421a1d060fad95a5a8b66ac79b49bff70fd722193175fda129c4` |
| Concurrent dense native probe aggregate receipt | `2b19929916df1e17ba8fc17bd1835760c72082aefbdcb123d42a48113e27956a` |
| Concurrent dense worker source patch | `2974fce19ea969062d9efcd6482f73b44771c72ee8478f0ea485a8b5e321915b` |
| Concurrent dense paired worker executable | `456fd309c3ddbeb0478b9f3eb483f865c348a50f5581b4cc3e710c8305b36d2c` |
| Concurrent dense worker proof executable | `9a786ea1cffd604fb7a4b7c95f0dbfac27cbe4c8eecf8b59f907b2feaaf10c25` |
| Concurrent dense worker and repeat evidence manifest | `6e7b1bb1e4015d8374f2cbaf5bdccae583ea71289357f4de5a3fb1862b678d5e` |
| Queued schedule comparison patch | `0215d1d603dd8957f5efd88ba1ff1be4a053bc94f877763935ef0fcfa5a9aaed` |
| Queued schedule comparison executable | `6b04b1a4dfe54aeb05abfd1cacf2f8dad014be394cb71eab4ecf55add1b51fe0` |
| Queued schedule comparison harness | `2e0b50d8a136c0450ffc5f50159912a7f6ccb4dc1d5750d64d1adfee734cbe11` |
| Queued schedule comparison aggregate receipt | `50d3f3f9e364819a7d199237a2110c4b0d6d2f3887070f1259c5c95ac652b223` |
| INT8 head bound and selected tile probe patch | `345fb4c094d1b3ed017e82130c49ef0e7eb5fb1b96dfc03803d4d35326c4c400` |
| INT8 head bound and selected tile probe executable | `3e91ce30d21769b79d2ba75f6353c9538dedf9aacf69e3d113fbcc4f9cbfeea7` |
| INT8 head bound and selected tile probe harness | `e613d0e3a6c869dd87fe98e3cc42233381a72276472b0b38e1760dfdc38b4b02` |
| INT8 head bound and selected tile aggregate receipt | `0898fc1264405331459791db6c40c14f5519d0bdd48ab9fa7046ef4b6d31af17` |
| Metal upper-mass follow-up patch | `95e393e211f0efdcecf8d84de7205d323947f09cd66af467e41211e288c4c7ab` |
| Metal upper-mass follow-up executable | `6bf4b48b6d9089dca05a288a24c707ae2cabfe559b1dc0e995ed958692f65a3a` |
| Metal upper-mass follow-up harness | `77367a595ba5bda001ab48550c51c2be04ecb611fe74da4108d99cbfa62eba50` |
| Metal upper-mass follow-up aggregate receipt | `d4b6e88f6cf6f70801483d7a6184ae311f0d218534e3de124ef315bc2752cdb1` |
| Queued sparse vocabulary readout probe patch | `7199ea67fb9b2a578365f47c614dec7da7c72f4ca527a0ea73b1b0e1720a8857` |
| Queued sparse vocabulary readout executable | `668420e44618ef4d5be5dbbea6f1465866502841702ad50341041b89e143cf84` |
| Queued sparse vocabulary readout harness | `ebf3ccc442ea46ce5c58f8bab5961e1f9cd9ec15f9d88cd8cf6b38885693c9c6` |
| Queued sparse vocabulary readout aggregate receipt | `e941eeb5d4aa13ffb15b2f2aa97d676532dd9768c229b0c8489f4510e0c3bf0e` |
