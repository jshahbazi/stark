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

## 20261122-INKDECODE-97 · Decode NVFP4 coefficients with scaled half bits

At `9b2491c7`, adapt the scaled E2M1 bit construction from MLX-Serve to the original Inkling checkpoint's adjacent nibble layout. Test pair masks, vector lane shifts, and pair masks with the power-of-two scale restored after the F32 block reduction. The last arm folds the recovery factor into the E4M3 weight scale. The source half ingress uses dyadic coefficients. The F32 block products and their sums stay within range under either scaling, so the existing dot grouping and scaled FMA remain in place. All three arms keep the same activation quantization, expert order and grouped epilogues. Compare against the retained 1 KiB FP16 pair table. [MLX-Serve source at `2fe9be83`](https://github.com/ddalcu/mlx-serve/blob/2fe9be834a601c70ef1a597563026340f0e722fe/src/kernels/moe/moe_fp4_e2m1x2_header.metal)

On the M5 Max, compare all 65,536 four-nibble words per candidate against the source coefficient values, including signed zero and the half subnormal that represents magnitude 0.5 before recovery. Restore the short and four stock 4K snapshots, with all four 4K initial logit rows matching pinned F32. Per fixture and candidate, rotate four profiled serial-logit checks per width at two through four rows, and 16 uninstrumented queued rounds per width at three and four rows using actual DSpark proposals. The queued comparisons include rejected suffixes. The candidates match every checked coefficient, verifier logit row, proposal and committed target and draft state.

| Candidate | Three-row NVFP4 dot change across 4K captures | Three-row paired queued saving | Four-row paired queued saving |
| --- | ---: | ---: | ---: |
| Pair masks, restore coefficients | +0.16 to +0.33 ms | −0.51 to +0.16 ms | −0.47 to +1.06 ms |
| Vector lane shifts, restore coefficients | +0.17 to +0.34 ms | −0.47 to +0.01 ms | −0.79 to −0.29 ms |
| Pair masks, restore at block scale | +0.03 to +0.21 ms | −0.36 to +0.18 ms | −0.32 to +0.36 ms |

Positive paired queued saving indicates lower wall time. In the separately instrumented runs the three-row NVFP4 dot stage increases on every 4K capture. The block-scale variant mixes small gains and losses in the real queued rounds. The short control changes by −0.05 ms at three rows and +0.03 ms at four rows for that arm. Archive the three shifts at the native gate and retain the table. The timing scope is the native queued round.

## 20261122-INKDECODE-98 · Check arithmetic guarantees for sparse vocabulary readout

Review the MPP products in the original BF16 vocabulary projection and the INT8 screening projection from attempt 96. A real-arithmetic bound follows from Cauchy–Schwarz. For a hidden row `h`, source row `w`, packed row `q` and scale `s`, the difference between the ideal dots is at most `||h||₂ · ||w − s·q||₂`. A runtime bound also needs the errors from both implemented products and the INT8 row scale, followed by an outward bound for the source cap. With ordinary round-to-nearest FP32 accumulation, the conservative accumulation factor at width 2,304 would be `2304·2⁻²⁴ / (1 − 2304·2⁻²⁴)`, approximately 0.0001374 per product against its absolute term sum. The reviewed Apple specification lists the supported input and output types and explains relaxed precision for float inputs. It supplies no accumulation error bound for the two MPP operations used here. Disabling the relaxed flag or replacing the coarse INT8 product alone leaves the original BF16 product's error assumption open. [Apple Metal Shading Language Specification, section 7.2.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf#page=354)

Keep the sparse path diagnostic. If testing it further, use the original compiled BF16 tile for selected scores with the production padded stride. Bound omitted mass from integer counts of omitted upper-logit bins. Use the FP32 difference between each bin's upper endpoint and the exact peak before bounding the production exponential, and carry the bounds outward through the FP64 sums and seeded comparison. Reuse the production dense sampler for ambiguous draws and resolve greedy signed-zero ties by token ID. This review establishes neither an exact sparse arithmetic contract nor worker throughput. A full original BF16 projection remains an available control for timing a sampler that avoids full-vocabulary ranking.

## 20261122-INKDECODE-99 · Separate draft ordering from sampled rank

At `9b2491c7`, rerun eight seeded original-checkpoint teacher trajectories on four stock 4K inputs using the INT8 DSpark shortlist. There are 192 anchors per trajectory and two conditional masks. The second mask's context uses the actual first target token. It earns acceptance only when the replayed first proposal would select that token. All eight initial F32 rows, 1,536 first-mask controls and 1,093 reachable second-mask controls match the pinned seed or the original native ranker as applicable.

For the first privileged arm, order the draft's four highest candidates by their actual target logits and assign them the draft's original mass at each rank. Apply a copy of the next uniform to this distribution, using the reordered first candidate if the draw falls outside the four masses. For the second privileged arm, read the full target distribution's sampled rank and choose that rank from the draft's original top four, falling back to its first candidate when the target rank is larger. Both arms use information available only after target verification. The top-four hindsight selector is a ceiling for the existing shortlist.

Fit a separate causal policy that predicts a draft rank from the mask index, the draft's top-two gap, the mass outside the top four and a bin for the copied uniform. Train on the other three prompts. Both seeds of the fourth prompt stay held out. Select four or eight draw bins and one of three smoothing weights by training round count. The fitted policy has no access to target values during held-out selection.

| Arm | Rounds across eight trajectories |
| --- | ---: |
| Stock top proposal | 769 |
| Correct target ordering within draft top four, original mass | 713 |
| Correct global target sampled rank, original draft order | 766 |
| Hindsight chooses the target if it appears in the draft top four | 665 |
| Held-out causal context bins | 771 |

The privileged ordering arm lowers round count on all eight trajectories, by 7.28% in aggregate. Supplying the global sampled rank alone changes it by 0.39%. The causal policy adds two rounds and improves neither seed on any held-out prompt. Archive the context bins. This screen supports investigating better candidate ordering if additional causal draft features are available. It records no live worker gain, and exports no token IDs, logits, draws or fitted parameters.

## 20261122-INKDECODE-100 · Fuse the original vocabulary readout and logit cap

At `9b2491c7`, keep the original BF16 weights, 16 × 32 Metal tensor descriptor and F32 output stride. Accumulate into a cooperative destination, apply the source FP32 soft cap to live rows, then store the tile. Compare four independent SIMD groups per threadgroup with the incumbent one, both with and without this epilogue. Keep the target's one-row path and DSpark's raw projection outside the change. The initial Metal load failed on an unavailable cooperative mask accessor and a missing standard-library include. The corrected variant uses the destination coordinates to restrict the cap to live, valid elements.

Run rotating pairs on the four-token short control and four stock 4K captures. All four 4K starting rows match pinned F32. With zero live capped rows, the three candidates compare 48,015,360 uncapped F32 words in total with the source projection, including padding, and match exactly. The profiled two- through four-row commands match serial capped logits. The uninstrumented queued pairs use 16 real proposal rounds at each of three and four rows per capture. All checked proposals, capped logits and committed target and draft states match.

| Candidate | Three-row readout GPU change across 4K captures | Three-row paired queued saving | Four-row paired queued saving |
| --- | ---: | ---: | ---: |
| Four SIMD groups, separate cap | −0.008 to +0.003 ms | −0.64 to +0.42 ms | −0.54 to +0.15 ms |
| One SIMD group, inline cap | −0.008 to +0.001 ms | −0.14 to +0.15 ms | −0.40 to +0.75 ms |
| Four SIMD groups, inline cap | −0.014 to +0.010 ms | −0.47 to +0.62 ms | −0.03 to +0.35 ms |

Positive paired queued saving indicates lower wall time. The original head uses 2.06–2.13 ms in the profiled three-row 4K commands. The inline cap and group layout change it by at most 0.014 ms, with mixed queued wall changes. Archive this layout at the native gate. The benchmark includes the original full vocabulary projection in every arm.

## 20261122-INKDECODE-101 · Fit a causal reranker from DSpark hidden features

At `94969405`, rerun the eight stock seeded trajectories from attempt 99. Record each native shortlist candidate's BF16 base score and folded Markov correction before their existing addition and soft cap. For the top four, add the previous target row's already-available score, a preceding-token repeat flag and the candidate's source weight RMS. The extended arm adds eight partial dot products between the corresponding normalized DSpark hidden row and the original BF16 vocabulary row. The same preceding target row is used for both masks. The second mask's features use the actual preceding token only on replay rounds where the first proposal would have matched it.

Fit ridge regression to the clipped differences between each of the six top-four candidate pairs' target logits, separately for the two masks. Hold out both seeds of a prompt together. For each fit, choose between stock and three blends of draft and fitted scores on the training prompts. Compare both selecting the highest reranked candidate and passing a copy of the next uniform through the original rank masses after reranking. Use the exact same sampled target trajectory and count disjoint rounds. Candidate features, logits, tokens and fitted weights stay in memory.

| Fit | Held-out rounds | Top-four target winner agreements with the unblended ordering |
| --- | ---: | ---: |
| Native DSpark | 769 | 2,284 / 3,072 |
| Base, Markov, preceding target score, repeat flag and weight RMS | 776 | 2,158 / 3,072 |
| Above with eight hidden projection groups | 776 | 2,153 / 3,072 |

All eight initial rows match pinned F32. The exported scorer reproduces all 1,536 first-mask native winners and all 1,093 second-mask winners reachable with stock. The prior reordered-target replay remains at 713 rounds. Archive both linear fits. Their held-out round count rises by seven, and target top-choice agreement also falls. The comparison does not time running the additional features in a worker.

## 20261122-INKDECODE-102 · Batch first-pass norms and share query and gate inputs

At `94969405`, batch the ordinary Q/K/V and expert RMS normalizations across verifier rows at visits without a cross-pass controller. For local attention, Q, K and V use the same input, so share the source 18-SIMD FP32 RMS reduction before applying the three separate BF16 weight scales. Global attention uses separate reductions when the three inputs differ. Keep the per-lane sums, inter-SIMD reduction, epsilon, inverse RMS and BF16 cast. The second arm projects query and attention gate in one grid over the same 3,072 output columns. This arm shares BF16 activation loads across both banks and keeps the original independent FP32 block and lane reduction order.

Restore the short snapshot and four original 4K snapshots. All four 4K starting rows match the pinned F32 reference. Four profiled rotations per width match serial F32 logits at two through four rows. Sixteen uninstrumented queued rotations per width and capture use real DSpark proposals at depths two and three. All checked proposals, F32 readouts and committed target and draft states agree.

| Candidate | Dispatches removed at three / four rows | Three-row paired queued saving across 4K | Four-row paired queued saving across 4K |
| --- | ---: | ---: | ---: |
| Batched first-pass norms | 240 / 336 | −0.036 to +0.561 ms | +0.515 to +1.314 ms |
| Paired query and gate | 48 / 48 | −0.310 to +0.032 ms | +0.069 to +0.373 ms |
| Both | 288 / 384 | +0.153 to +0.600 ms | −0.218 to +1.199 ms |

Positive savings indicate lower wall time. The norm arm reduces the three-row instrumented attention input stage by 0.350–0.364 ms and the expert input stage by 0.069–0.121 ms on the same captures. The live comparison advances both changes at three rows and uses only batched normalization at four and at the two-row tail.

### Same-loaded worker comparison

Alternate the retained path and the width-specific candidate in one loaded M5 Max worker. Vision and the native INT8 DSpark head stay resident. Every request takes 4,096 cold inputs, uses context 4,352, reuses zero inputs and samples and emits 256 tokens. Greedy uses three proposals and seeded temperature-one, top-p-one sampling uses two. All 16 output hashes match the pinned reference. Corresponding proposals, acceptance counts and round histograms match. The sampled arms each use 530 rounds and the greedy arms each use 398.

| Mode | Control median | Candidate median | Candidate range | Median paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 82.03 tok/s | 83.12 tok/s | 80.32–86.82 | +2.32% | 4/4 |
| Seeded sampled | 67.89 tok/s | 67.52 tok/s | 64.50–69.82 | −0.29% | 1/4 |

The greedy paired median saves 0.735 ms per round. The sampled paired median adds 0.085 ms. The worker accounts for 99.90% of observed Metal counter increments across 44 request-interior intervals, with nominal and fair thermal states and unchanged pageout and swapout counts. Greedy verifies four rows in steady state and uses batched norms there. Its short tail can exercise the other layouts.

Reverse each pair and repeat both modes on the same executable in a newly loaded worker. All 16 pinned hashes and paired work counters match again. The greedy control and candidate medians are 83.20 and 83.21 tok/s, a +0.38% paired median with three faster pairs. Every greedy candidate remains above 80, spanning 80.45–85.30 tok/s. Sampled medians are 68.81 and 68.33 tok/s, a +0.47% paired median with two faster pairs. The worker accounts for 99.89% of Metal counter increments across 45 request-interior intervals with fair thermal readings and unchanged pageout and swapout counts.

Across both comparisons, the greedy paired median is +1.12% with seven of eight wins. The sampled paired median is −0.29% with three of eight wins. Advance only four-row first-pass norm batching on the existing batched residual verifier. Keep the existing two- and three-row kernels, including the sampled steady-state path. The paired query and gate projection stays in the native experiment.

### Narrowed verifier

Register the two batched RMS kernels with the existing opt-in residual mixer. The retained source selects them only at four rows on non-controller visits. No diagnostic arm switch or query/gate kernel is present in this build. The flags-unset native build reproduces the connected short and 4K slices, the pinned short full-model F32 row, and all 16 stock F32 rows, 768 BF16 residual comparisons and four snapshot replays. With the fast verifier enabled it repeats all 45 rejection and ring-wrap checks and the 96-token serial continuation and state checks at each proposal depth on short and both stock 4K captures.

The retained worker preserves exact 2,048-token prefix reuse across proposal depths, clears reuse on a source-mode change or cancellation, and passes short and long-context admission and accepted-proposal stop accounting. A separate cold 4K observation uses the same eight pinned requests and matches every output hash. The medians are **84.09 tok/s greedy** (78.68–90.60, three of four at or above 80) and **66.99 tok/s seeded sampled** (62.84–69.29). The worker accounts for 99.85% of Metal counter increments across 35 request-interior intervals with fair thermal readings and unchanged pageout and swapout counts. Use the same-loaded comparisons above for relative speed.

## 20261122-INKDECODE-103 · Reorder exact dense weights for coalesced verifier reads

At `5d1a7767`, reorder the original BF16 query, key, value, relative, attention gate and attention output matrices. Pair adjacent output columns so the two halves of each SIMD group read one contiguous stripe. A second layout also places adjacent terms for each lane together so one 16-byte load supplies two BF16 vectors. Both layouts preserve the source dot grouping, final shuffle reduction and BF16 rounding. Check every rearranged vector against its original bits. The two layouts hold 144 banks each and allocate 2,576,351,232 bytes together.

Rotate the three layouts on the short and four stock 4K captures. All four 4K boundaries match the pinned F32 row. Across four profiled rotations per width, all 540 full F32 readouts match serial. Across 16 queued rotations at each of three and four rows per capture, all proposals, verifier logits and 320 candidate committed states agree.

| Layout | Three-row dense GPU saving across 4K | Three-row paired queued saving | Four-row paired queued saving |
| --- | ---: | ---: | ---: |
| Adjacent-output stripes | +0.034 to +0.048 ms | −0.148 to +0.237 ms | −0.782 to +0.517 ms |
| Adjacent-output stripes with 16-byte lane loads | −0.015 to +0.046 ms | −0.325 to +0.496 ms | −0.925 to +0.545 ms |

Positive savings indicate lower wall or GPU time. The sampled-width dense stage uses about 5.7 ms in the control. The permutations change that stage by under 0.05 ms, and the complete queued effects vary by capture. Archive both layouts at the native gate.

## 20261122-INKDECODE-104 · Share DSpark attention reads across its draft block

At `5d1a7767`, keep all eight rows of the DSpark block and the original shard boundaries. Group two or four query rows under one committed K/V load, group the three attention heads that already share one K/V head, or combine the two-row and three-head layouts. Per query, retain the same visit order, FP32 dot, running maximum, weighted sum and BF16 join.

Run the short and four stock 4K fixtures with the original checkpoint and native INT8 draft head. Each stock initial row matches pinned F32. At 16 consecutive anchors per depth and fixture, compare each geometry against stock both as an isolated proposal command and queued with the actual target verifier. All 3,686,400 checked BF16 draft hidden words, proposals, provisional target logits and 640 candidate commits agree, including rejected suffixes.

| Shared queries | Isolated two-proposal savings across 4K | Paired queued savings across 4K |
| --- | ---: | ---: |
| Two rows | +0.132 to +0.853 ms | −0.407 to +0.484 ms |
| Four rows | +0.473 to +0.591 ms | −0.097 to +0.373 ms |
| Three heads | +0.307 to +0.565 ms | −0.200 to +0.055 ms |
| Two rows and three heads | +0.358 to +0.624 ms | −0.296 to +0.245 ms |

The four-row variant is the most consistent isolated two-proposal gain. At depth three its queued comparisons all regress by 0.090–0.389 ms. At depth two, no geometry improves the complete queued round on all four captures, and the best paired median is under 0.16 ms. Keep the isolated measurements and archive the layouts at the queued gate.

## 20261122-INKDECODE-105 · Compute exact FP4 pair dots with packed multiplication

At `5d1a7767`, preserve the four-over-six activation selection and the source K16 accumulation. Map each FP4 coefficient to twice its signed value, then add 12 to get a digit in 0–24. Pack the activation pair in base 2,048 and the weight pair in reverse order. One integer product then gives their unsigned two-term dot in its middle 11 bits. The low digit is at most 576 and the middle digit at most 1,152, so neither affects the next digit by carry. Subtract the original bias contributions and recover the source FP32 activation and weight scales before the existing scaled FMA. The weight table uses 1 KiB, and each K16 activation occupies 24 bytes.

On the M5 Max, check all 65,536 activation and weight byte pairs against the signed integer dot. The short capture and four stock 4K captures match all checked F32 verifier rows and the paired queued proposals and committed states. The initial 4K logits match pinned F32. Rotate four profiled comparisons at widths two through four and 16 uninstrumented queued comparisons at three and four rows per fixture.

At three rows, the packed multiply adds **0.047–0.109 ms** to the measured NVFP4 dot stage and **0.290–0.321 ms** to activation quantization across 4K captures. The paired queued savings span **−0.741 to +0.315 ms** at three rows and **−0.539 to +0.212 ms** at four. The smaller table avoids most of the earlier lookup's regression. Packing the live activations widens the remaining regression. Archive the candidate at the native gate.

## 20261122-INKDECODE-106 · Combine the strongest remaining sampled-width changes

At `5d1a7767`, combine three-row first-pass RMS batching from attempt 102, three-head local K/V sharing in the target from attempt 91, and four-query-row K/V sharing in DSpark from attempt 104. The control uses the existing three-row RMS and DSpark paths, with the target local kernel recompiled in the generalized shader at its incumbent one-head, two-row grouping. Both arms use the production target sampler.

Rotate both arms at 16 consecutive anchors on short and four stock 4K captures, at proposal depths two and three. All stock initial F32 rows, checked draft hidden words, proposals, provisional F32 target logits and committed target and draft states match. At the sampled width, the short queued savings are 0.480 ms control first and 0.469 ms candidate first. On the stock 4K captures, savings range from −0.302 to +0.024 ms with control first and +1.516 to +2.114 ms with candidate first. The isolated draft measurements also depend on position. Use the loaded worker with reversed pairs to judge speed.

The first worker build used the unqualified name of the sampling enum. The timed build selects the probe only for a stochastic depth-two request and resets it when a request falls outside that scope.

### Cold sampled worker pairs

Run four pinned 4K sampled requests in both arms, then reverse the workload order and each arm order in the same loaded worker. Use 4,096 cold inputs with no reuse, context 4,352, the original checkpoint with vision and INT8 DSpark resident, two proposals, and exactly 256 sampled and emitted tokens. All 16 pinned output hashes, accepted and proposed counts and round histograms match. Both arms execute 1,060 rounds.

| Pass | Control median | Candidate median | Median paired gain | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| First order | 63.53 tok/s | 66.41 tok/s | +2.86% | 3/4 |
| Reversed | 56.19 tok/s | 58.82 tok/s | +3.07% | 4/4 |
| Both passes | 62.56 tok/s | 62.61 tok/s | +2.86% | 7/8 |

The median paired useful-time saving is **0.896 ms per round**. Absolute speed drifts across the session, and the fastest candidate request is 70.98 tok/s. The worker accounts for 99.84% of observed Metal counter increments across 58 request-interior intervals. Thermal readings are nominal or fair, with unchanged pageout and swapout counts. Compare within each pair to judge the effect. The prior retained worker needs a median 4.73 ms less per round to put each of these workload budgets at 80 tok/s at its measured acceptance.

### Three-row verifier with a generalized local shader

Remove the worker A/B selector. Admit the changes under the existing opt-in residual mixer and shared local attention. The target uses batched norms and three-head local attention only at three rows. DSpark shares K/V loads only for two live proposals in source-text mode. This includes a two-proposal tail of a longer request. The kernels use the existing snapshot format and sampler state.

The flags-unset build reproduces connected short and 4K parity, the pinned short F32 row, and the stock 16 F32 vectors, 768 BF16 residual comparisons and four snapshot replays. The enabled verifier also passes all 45 rejected-suffix and local-ring controls and matches the 96-token serial continuation and committed state at every proposal depth on short and both stock 4K inputs.

All eight cold 4K worker hashes match on the retained build. It preserves exact 2,048-token prefix reuse across proposal depths, clears reuse on mode changes or cancellation, and passes admission and accepted-proposal stop accounting. Its median rates in this observation are **77.81 tok/s greedy** (67.49–83.80) and **62.85 tok/s seeded sampled** (60.15–68.89). Prefill spans 723–843 tok/s. The worker accounts for 99.77% of Metal counter increments across 39 request-interior intervals with fair thermals, and pageouts and swapouts are unchanged.

Recheck the previous sealed worker and then the generalized-shader worker with the same eight requests. The previous build scores 80.95 greedy and 67.20 sampled at nominal thermals. The generalized-shader build scores 79.56 greedy and 67.84 sampled at fair thermals. Median prefill is 948.38 and 929.96 tok/s respectively. All 16 output hashes match. These are separate loaded-worker observations.

### Final shader selection

Keep the incumbent local-attention shader source, registration and dispatch for widths outside the three-row path. Register a separate three-head shader for the fast three-row verifier. It writes the same per-row, head and shard parts and reads pre-write K/V for displaced positions. Retain first-pass norm batching and DSpark sharing under the same verified widths. The first build of the separate shader variant ran out of disk space during compilation. Reclaim the previous probe's reproducible Cargo cache and build again.

The final native executable repeats the flags-unset short and stock 4K pinned controls and all 45 enabled rejection and ring-wrap checks. At each proposal depth it matches the 96-token serial continuation and committed state.

The final worker matches all eight pinned cold 4K output hashes. It repeats lifecycle checks for exact 2,048-token prefix reuse across proposal depths, cancellation and mode changes, admission and accepted-proposal stops. In its separate bench run, median throughput is **82.91 tok/s greedy** (79.14–86.05, three of four at or above 80) and **69.37 tok/s seeded sampled** (67.22–71.48). The worker accounts for 99.87% of observed Metal counter increments across 35 intervals at fair thermals with zero pageout or swapout deltas. At the four sampled requests' measured acceptance, reaching 80 still requires a median **3.77 ms less per round**. Retain the exact three-row geometry under the existing opt-in verifier configuration.

## 20261122-INKDECODE-107 · Reuse DSpark mask rows across verifier rounds

At `789f8efa`, use the retained short verifier, the original target and the bundled DSpark with its INT8 vocabulary projection. MLX-Serve separates the draft block projection from its conditional Markov step. Test retaining seven normalized mask rows, their projected scores and the refresh anchor's feature. Rebuild the Markov state using only emitted IDs when a later round reuses the rows. Submit proposals and the same two-proposal target verifier on the ordinary queue. Refresh when the live masks run out. [MLX-Serve draft implementation](https://github.com/ddalcu/mlx-serve/blob/2fe9be834a601c70ef1a597563026340f0e722fe/src/plugins/inkling2/draft.zig#L195-L282)

Use the short fixture and four stock 4K captures with two temperature-one seeds per 4K prefix. Each 4K arm starts from the same restored state and advances over 128 timed outputs from an exhaustive seeded serial replay. Reverse the policy order for the alternate seed. Measure total queued and commit time including any extra rounds. All four starting F32 rows match the pinned captures. All 4,288 consumed F32 rows and 36 final target and draft state comparisons across short and 4K match serial.

| Refresh policy | 4K rounds | Accepted proposals | Reused rounds | Median relative queued plus commit time | Faster trajectories |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh each round | 541 | 483 | 0 | 1.000× | — |
| Every two rounds | 651 | 373 | 320 | 1.163× | 0/8 |
| Every three rounds | 729 | 295 | 475 | 1.274× | 0/8 |
| Every three rounds or after a rejection | 590 | 434 | 123 | 1.058× | 0/8 |

The two-round policy's reused rounds average 2.62 ms less queued time than its fresh rounds at the median across trajectories. They run at different positions. One short arm improved, and even the rejection-sensitive policy needed 49 extra 4K rounds. Archive the frozen-mask policies at the native gate. The timing excludes seeded sampling and the diagnostic's full-row equality checks.

## 20261122-INKDECODE-108 · Reduce NVFP4 down routes inside one threadgroup

At `789f8efa`, test a three-row verifier layout from MLX-Serve's packed down projection. Keep the original half ingress and its route-weight placement. Scatter the weighted gate/up activation in original route order before down quantization. Assign eight SIMD groups to the eight picks of one output channel for one token. Each group reproduces the source K16 dot, scaled FMA and BF16 rounding for its pick. Sum the eight rounded contributions in order and pass the merged BF16 row to the existing convolution. This removes the per-pick dot and epilogue intermediates and gives up weight sharing across tokens in the down product. [MLX-Serve down layout](https://github.com/ddalcu/mlx-serve/blob/2fe9be834a601c70ef1a597563026340f0e722fe/src/plugins/inkling2/kernels/downs.metal#L20-L46) · [Earlier down epilogue fusion](#20261122-inkdecode-87--fuse-nvfp4-expert-buffer-boundaries)

Use the original checkpoint and retained queued INT8 draft on the short and four stock 4K captures. Compare both arms at 16 restored anchors per capture and rotate one, two and three kept rows. Across three runs, the candidate matches 720 full F32 rows, all proposed IDs and 240 target and draft state pairs. The first verified row also matches singleton decode at every anchor.

The first alternating run appears to save 1.265–2.953 ms at 4K by paired medians. Split by run order, the candidate adds 6.460–7.681 ms with the control first and saves 5.561–6.715 ms with the candidate first. A repeat with four commands per anchor still shows position effects. Run both four-command orders at each anchor against the same snapshot, and take the median of the difference between each arm's four-command mean.

| Fixture | Median queued saving | Mean queued saving | Faster anchors |
| --- | ---: | ---: | ---: |
| Short | −0.392 ms | −0.397 ms | 0/16 |
| Sampled 4K A | −0.665 ms | −0.497 ms | 8/16 |
| Sampled 4K B | −0.498 ms | −0.443 ms | 8/16 |
| Greedy 4K A | −0.489 ms | −0.336 ms | 8/16 |
| Greedy 4K B | −0.712 ms | −0.328 ms | 7/16 |

The layout adds time at every fixture median after both orders run at each anchor. Archive it at the native gate. The timing covers the queued proposal and verifier at depth two, including on the prompts originally captured for greedy use.

## 20261122-INKDECODE-109 · Expand the NVFP4 coefficients before verification

At `addddcff`, replace the live FP4 byte-pair lookup with loads from one of two lossless layouts on the M5 Max. Store each coefficient as binary16 or store its high byte and shift it into the original binary16 bit pattern on load. Both preserve signed zero, the original E4M3 scale bytes, the per-lane grouped F32 dot and the scaled FMA and SIMD reduction. Expand both expert banks of all 20 physical NVFP4 layers on the GPU after restoring the prompt seed. Select the layout only in the three-row verifier. The extra coefficient buffers use 18.98 GiB for the high byte and 37.97 GiB for binary16 while keeping the reference banks available.

Use the short and two stock sampled 4K captures, the original checkpoint, and the retained INT8 draft. Compare raw up and down products in a separate shadow pass over eight restored anchors per layout and fixture. All 233,570,304 F32 product words, 144 full F32 logit rows, draft IDs and 48 committed state pairs match. Time the plain queued commands in both orders at each anchor. Collect four separate timestamped verifier rounds per layout and its packed control to attribute the dot-stage cost.

| Capture | High-byte queued increase | Binary16 queued increase | High-byte dot time, packed → expanded | Binary16 dot time, packed → expanded |
| --- | ---: | ---: | ---: | ---: |
| Short | 4.776 ms | 15.447 ms | 6.106 → 10.608 ms | 6.169 → 19.565 ms |
| Sampled 4K A | 4.453 ms | 14.398 ms | 6.086 → 10.310 ms | 5.937 → 18.172 ms |
| Sampled 4K B | 4.952 ms | 14.573 ms | 6.220 → 10.696 ms | 6.466 → 20.385 ms |

The first build needed a corrected allocation-size conversion. Both Metal variants then needed explicit vector casts. An interruption cut the corrected run after the first three receipts, and the remaining cells resumed with the same executable. The short binary16 cell observed 11 pageouts and the first 4K high-byte cell observed 16,416 swapouts between host snapshots. The other four showed neither delta, including both second 4K layout cells. Both layouts regress at the native gate even in those cells. Archive the expansion; it never reached the worker gate.

## 20261122-INKDECODE-110 · Shorten the DSpark forward

At `addddcff`, test all nonempty subsets of the three original DSpark layers on the M5 Max. Run each subset in the original order with all eight input rows, the original final RMS, the same INT8 vocabulary head and the Markov step. Keep all three draft cache updates and the ordinary target verifier. This changes the proposal model while preserving the target's seeded draw.

Use the short control and four stock 4K captures. At each 4K capture, replay two seeds at temperature one for 128 timed outputs. Generate the serial target draws with the full sorted vocabulary. At each anchor, rotate the seven draft layer subsets and record accepted proposal prefixes and native proposal time. Count disjoint rounds with a maximum of two proposals and the ordinary output-limit cap. At every sixteenth anchor of the first seed, replay the seven arms in both orders on one snapshot to measure queued wall time. The initial 4K rows match pinned F32. Every timed queue arm matches its isolated proposal IDs, the other arms' full anchor logits and their state after committing the anchor.

| Executed draft layers | 4K replay rounds | First / two accepted prefixes | Paired queued saving by capture | Modeled 4K time ratio | Faster trajectories |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, 2, 3 | 541 | 669 / 353 | — | 1.000× | — |
| 1, 2 | 794 | 243 / 53 | −0.048 to +1.090 ms | 1.331× | 0/8 |
| 1, 3 | 589 | 570 / 176 | −0.947 to +1.035 ms | 1.049× | 2/8 |
| 2, 3 | 977 | 48 / 2 | −1.229 to +0.868 ms | 1.905× | 0/8 |
| 1 | 919 | 109 / 10 | +0.577 to +1.827 ms | 1.600× | 0/8 |
| 2 | 956 | 69 / 4 | −0.691 to +1.664 ms | 1.744× | 0/8 |
| 3 | 922 | 101 / 3 | +0.144 to +1.446 ms | 1.726× | 0/8 |

For each fixture, price both seeds with the first seed's median full-draft queue time and the paired queue-time difference. The modeled time omits host sampling and commit time and uses a fixed cost across round depths. The first-and-last-layer subset is closest to the original model. It adds 48 rounds across the eight trajectories, and its median modeled time rises 4.9%. Archive the six untrained subsets at the native gate.

## Evidence

The original target and DSpark artifact digests are recorded with the [integrated QKV run](2026-11-21-inkling-mini-decode-60.md#evidence-identities). The worker and native screen retain aggregate-only private receipts. No prompts, token IDs, logits, targets or fitted tensors are published.

The first source exports for attempts 97, 99 and 100 used Git's configured external diff renderer. The original visual diff digests and executable identities remain recorded. The evidence table also gives the applyable Git exports of those same private source trees.

| Evidence | SHA-256 |
| --- | --- |
| DSpark layer ablation source patch, executable, driver and summarizer | `82c960bb118d3a15d7bcba757a30c9947170920c8520c8ea51ab16629e896a82`, `9c6030f3cad5da85f7b3302fab2cc04764e18eaf64ad4094d67cc8979cfa16e2`, `8cfa5ef6a3b667cb2eb13612c538d50b8f1c8731169eafde9ae6b4321e168279`, `3701e546069a4dcc02414cd2ae8a839bc07e8268a32a5f22a808e497d52c2df0` |
| DSpark layer ablation short and 4K receipts | `fcd1db6c45bf3524f26cc5dd0a0bb8599cb05c37d2dcd37760249830a3317516`, `67eff3f71b9a2426d8422820aeeab42d1ddb4dcb3a1d8342bd9e5f97b768a88a`, `8a7ed39c4693cb9b8d396a0f843b4b681eed1af8e9ffec6c38289b44a951f344`, `e8f44944a07b2391aa3cef796a16da98b9f6032530e83a4ed5891d63331a42f9`, `d77183fcff2cdb6cf18dafdc8ca0cc9e7db1a9d750a39d6e2953bbcf13231a1f` |
| Expanded NVFP4 source patch, executable, driver and resume driver | `6920f48e61a56e45f8912bafbfe47cae8c6454c478677de57c1a9669ba8acf8d`, `3b7e6e7df34f75b716bc55b1a89aea4bb4b5a00d2d6cc966ee76a042be622a77`, `0f02a0f9364561a1722614f920a9a17fc9e615bbdb629674c5bd4378ed7682eb`, `f18f97ac7f5a8d5652bce63ee1c206586622e5596a7e4b05a13e694305356e88` |
| Expanded NVFP4 short and sampled 4K receipt pairs | `11d15f7f002b7dfebe085a919f92d3db37f4ead96c76ac17c5943488e27cc25a`, `d430ff38a64f370d52af3de33d18e2a1f06906d4c773ab740cfd2b113e545a69`, `55d1902dfb17bee158c2b9400f14a106deb141071d5c34bbd9122f93a548782c`, `2e66b2515e9acbe6dc0982059520a0bac0bb4457e33993dbf5ce4e7223cf3c17`, `47b7dcf3b7f52b5ce8812ab6067080c19c4b9f8d0e8f289cd336515865d87c86`, `fe8770a1201919e73863542898a7c1b8975763a39ffd961a9218c8b2f913cb16` |
| DSpark mask reuse Git source patch, native executable and driver | `1a19f74e23580e70d89b70a3c7456d294eef4ce404dbb0c3840d9616067f3e35`, `ce93b1efde9085ed3e07aa213e916cde8e4f1cf58af2ed062c3a43cbe4330c80`, `ae21abb8dfd59765baddd29e4221a67a97ecde21b8a21592ab0018a8a384eeee` |
| DSpark mask reuse short and 4K receipts | `5242668592527813e50f7406e0819a39a7b2274788bc57066796941b7e72b562`, `2316ba0f8977edb989dbd0e4a1b49dc3bbbcf1a762acbef6c7e768da8782a0ce`, `d2865d0621678b21d6c41f1d52ad83806414326e2a65c0fa89a2e17d3129379b`, `6a220626b82e480d9a9ec4d8e0f25cfa9807e9ece52aeb6b945e8a32374b5c9b`, `e0e569eb560bc3d73fbf94cec459bb4ca51dd7088c1c9321c06738aa028f4328` |
| Per-token down initial, four-command, and final Git source patches | `ad09e67106d7f44fb5411efb06282eae3efef35452ebf5b97f8c6732764a8fe7`, `b7c913b8cc709987a681bd9155544528297e69f60c1523cc68e1ff0a4a79b75c`, `9179ce64ab7467a500aa64adee4ad5b17f60d382031d2b3626a2928ef8f6a3bb` |
| Per-token down initial, four-command, and final native executables | `a2ef68315d416304d8ca785fd9041758be0fb35aab0629af26d6c1e8ea1ede3b`, `a2a7edadf313d9dfcb1cb5d6ac46f2cd1e205aac22520ddb3e398df7307067a5`, `9ba69bbc3f7b4f9d0dae901af32968c4e2dac81390fe31496d4a91ed451690f6` |
| Per-token down final native driver | `a18ad32e094eeef73986275e7d74f1348b153c14c376926106edc0faa383ca37` |
| Per-token down final short and 4K receipts | `22651a5c96d843339965f52e673b2d12a15560ca9cb8e285107d676610895a07`, `8100eb578fdb77fd2e1bb23ccab9cb523da9f2ec222f00b02c7e6e9e4280f9d4`, `33fa2279794b6d21fa2785941debb5f237d69fc53cf7872eeb9c59ee1ffc8166`, `aa54d5a8829ccb44e02f646ab153590b308c46e74d5c9c57271893988b228af6`, `7df977113ce7e7db4b434c371c8c2893e42f8f1920a8658414c6b5529028e395` |
| Sealed integrated worker | `638741dc417d72591a96b3fe5f4bccac64b345950319c09bfcded6d11100b485` |
| NVFP4 shifts visual source diff | `cf5664ddfac2ff8c2fd0965faadee27eae5a72efdd960e8d6211eb6f33b200d5` |
| NVFP4 shifts Git source patch | `05fb0bec7032ba589639f8add918e345db9ae0b658774164b15718a8f43726e1` |
| NVFP4 shifts executable | `75fe9a910100f540ebd0912106d59fc07e6c6b6883d1be615ac7c692a18a712d` |
| NVFP4 shifts harness | `ebef89e8e0a9aca3c880793e514537a60cfa46476e71e681f3c207df2b79ece6` |
| Pinned MLX-Serve MIT license | `dfe22e0f100fe7f2c2808f1af6f488e7339cf8deb7504153602b4169ed7d90b4` |
| NVFP4 shifts short receipt | `7e575cd8d39d30474c7f4c1ffb06b6315639293783b3d4dfb418df78264dc3d8` |
| NVFP4 shifts stock receipts | `9a1ec887d30e5b7839ad04ee653701b6dee8473e55761fc93960876229460f15`, `508e8835a829e62c06e0715020e3dec3cc0995d33abbd0ac364c73e8c6679890`, `390f0f3b55d770a2d538f8181347f3b97330dc0122b3a9524d40e33f46ea3a09`, `b83355f281acc4895ec984be2c2964a3d42fde241b2610092269caa48b168c4c` |
| Draft ordering replay visual source diff | `2f1e3334c77fa1a4fda7f2cb4c6d567e72775ec0f9d9a8dc9c1959541f392124` |
| Draft ordering replay Git source patch | `e4a10c511c309eda09cf31a8fb419ef8883cc115f03cadb82aeaabf9bf2a4df8` |
| Draft ordering replay executable | `d267cad9ce87268ae9da35cc9a7fac1827660ba4e49cf3cd654eaddd57350a60` |
| Draft ordering replay harness | `b62b17f266eb352813d69405a1cfbb27796ea17fe8dac079b8ac6b03c2d1ec5c` |
| Draft ordering replay receipt | `1fe0ff9a27792df4b83f3377c840141e4a4bcf81d84e45e5678f9005dea6ff3a` |
| Fused vocabulary readout visual source diff | `355517c19b6e538bb19bf3259e9d91eef94e5fe8f2526531fecfb55be193732f` |
| Fused vocabulary readout Git source patch | `7bcbd2fc70301b330c65aa7d111d83c514d53889810386a623d74c83fe9f64fd` |
| Fused vocabulary readout executable | `ee2187be5d7c85777c789e86b5f80fd55ba142eda15870a814ec81b40eeede23` |
| Fused vocabulary readout harness | `5d5a34f0fb66c61f15742ed52e686c7538bab8488bce2d200280ae91434a9253` |
| Fused vocabulary readout short receipt | `7f64491b92fd0263e7354c5f9bdc09bf5463d65b852d6ee607a648a3f85f190a` |
| Fused vocabulary readout stock receipts | `ee2024e62070d603c8631ae2c5106e2b6739597fa57de9509edc50edddf79e48`, `b2d191e1c897f806f8ffaa52cd390fde022853a8a0a80422e3f9aeca373f980f`, `32ad833396bcc41048033e9f91bb74d94dba365798446dda9380d1b82adec9e1`, `08004219e41dc22e8a1aabad3988a256b5f23686fbb1b3aeddfdffddeb5d51f5` |
| DSpark hidden feature Git source patch | `f72fc8ddb98c0f11fb1a1b113ee754fcfc2a39fba16b02676ed17a70f3eda8db` |
| DSpark hidden feature executable | `0a422537f63bf1858e538d4122a3492a285cb52556421a657551ec8eeef40b8a` |
| DSpark hidden feature harness | `de7192dd664a659af704096561ad501bdefd58febfd9838be8cff9d53dc4fb67` |
| DSpark hidden feature receipt | `c3d5d12e0bd5bf98a0c2c0c8b6e965361116ed6290173cdb86cac180bdd3c821` |
| First-pass norm and query/gate native Git source patch | `a48b73a15c3c23c62928e99e06fad02a02c1cb100afe1193b51aa1513ce9b1c2` |
| First-pass norm and query/gate native executable | `442f89c1d6d96c3fc0759c170b662385fb38a353d7c341870bfb6ef74142bbb1` |
| First-pass norm and query/gate native harness | `2d58bb3ba8c5198d122d4c4dca1a24c3220ac91c2f7015dace5e00df0cb452cf` |
| First-pass norm and query/gate short receipt | `95e1a3d08b50a6314cd1959495dcceb056b9cdf66ca52c5400e729dee06d1c69` |
| First-pass norm and query/gate stock receipts | `892aac61704acc8501ba7aa1ec23be33a5f2bd47bc5d69fe427803933b68262c`, `6cf050c33926a6f22dcd4d445d162d5d06e8ae97602cb96f1ddca6a8493dedf7`, `f6b70172b6231a7a24b71ded150db74d0e3f1903044d1d22d58097c4dc59a5b1`, `b7d6e5754ce4f6e268b2af819d64585d6ef84011b3b3947d822585daeae08ce3` |
| First-pass norm and query/gate worker Git source patch | `a27dabbd4e6961b9aa315da655432078dcb0b29107dd51e219e67364757817f2` |
| First-pass norm and query/gate worker executable | `7ec03c9d11a6788ce0be56f0cb5cb9b15f6d56a375376b37465af37fd42828e3` |
| First-pass norm and query/gate worker driver and scorer | `2dfef7b64c6daca0e73e8a09463e7614015c0172b324ffcdc33adb0c1ab9137f`, `b4b54ed529d64d9a7d74775b367caae9b3f876336e34fc7e2f2c0c8083ad394b` |
| First-pass norm and query/gate worker pairs | `ecbf8abd4cc49957c9c40ec995e99e64a924c0131771845b952487996c7a3f48` |
| First-pass norm and query/gate host observations | `c1d8cdb3412279316d58a8681d2bcfec59392ceab8013672ef52197ae43bd87a` |
| First-pass reverse repeat driver and scorer | `be015ac6f743357076fcf4f745d39913b2167e32d9526bb41be1dabc0442878d`, `ca921ba3218488129b8f67949389bf04d52decdf93a1c10fd034fd498cc6a9a9` |
| First-pass reverse repeat pairs | `e8388fca6cd179f897178458cec1793ab65ae8aac00a7d0b08540c8f847c919d` |
| First-pass reverse repeat host observations | `ae1cdc90bd45285d4e9f8cca6d5d889f74a2845cc4685d6d0638cdcca7a689be` |
| Four-row norm Git source patch | `666c057718c0f878b5a572fd8bbe94eabdba51b289573b8c456b334ea2845af9` |
| Four-row norm native executable | `1345dad55e285d26fe35dc449b31d8a7890af5a6ef7a39440ae2a6c6d6c05e3e` |
| Four-row norm worker executable | `e90b71f2740256de3af27db2e707d9831693584ed3a651640da9d31e203ba94e` |
| Four-row norm qualification driver | `46cc2e6cf6b0b7b268b7233b13da5b0e188d87a70b6d7d38f68992f427c1a9b2` |
| Four-row norm flags-unset summary | `b8a5b540c43b3942a458316f8d3f2943ed64beadfbb40fe803a93111aac3d011` |
| Four-row norm enabled proof receipts | `96ffe7905c01610297455fb406e1b19206a6901434ee81b0f3653f3a8ece44ca`, `be40ac1c68778fb924a7ca79984e4a371522764f8dc9bf2bd7c664106a7bba3e`, `d9d3bd833801bed6a74472f2b7425e496da9d55ee869c9fd508362ae62d67a81` |
| Four-row norm lifecycle receipt | `d1b51cd09652bd043eb12d6546e89535af29ed286b5f587400a88200975afc6b` |
| Four-row norm stock 4K worker results | `0bccab39fd8a3ce9f564f5c3921c902d2697244114c794db9e0a99350689db43` |
| Four-row norm worker host observations | `ef78cba7143095c72a893ffd174a9df3af732dbe1a8aeebc5d1d31f08a757eb3` |
| Dense weight panel Git source patch | `e6bfab17d075d01588edd2c035b3d99cc135a4e2f249d8ab366a4fec620484bd` |
| Dense weight panel executable and driver | `50b5067ff9f6073a77f51686c9a0afc9f6d60e6561f2d74cf94df794634e940b`, `dab7e32cc071ea62382609068c3ae3b7d50f0803173592e8ffc153683fc9a2da` |
| Dense weight panel short receipt | `4ec4c857c9ef32c4679c710a5bb7411125aff5307be93d508929c26791248f9c` |
| Dense weight panel stock receipts | `dac81a1586785ae964df8ac37ff9d4edd89bad924f5261841b83a5155382d709`, `8e4237eba579140da791c0c10bfb491bff74d201f8e15bdac4769a768f9600af`, `e67c66c12badf8724ed3c117939a0db07016ff50879b9155772653bea202d7b2`, `02a0be9baa53126148adc8ad4706986021319ab9df4aa225ac7f1ce6009e267a` |
| DSpark K/V sharing Git source patch | `aa3199a6c6873aa1fe157e3841a4735752591f6aa2a78d2115d0a9a5e3c9d530` |
| DSpark K/V sharing executable and driver | `9368a8e5a86fff54aeb4a878c47012a6a6df60894dfe67860863057f953905cf`, `e9a2fb750ff60be502726c9538e615a550b07676ab5069a01fd1526f93300358` |
| DSpark K/V sharing short receipt | `e6f5e5f0b5149a271d00a596eb29fe211c89cd9877d02798a90aedcfbdb761a4` |
| DSpark K/V sharing stock receipts | `a1931fbdf878cadba1da7990b9a2dae842a3e726361a0808b612acee2c358450`, `c1ef10af4358c3ea273e65f000cb7f388441e97007a6b8f9301bfe58d7a08f64`, `448f4fd2f2b06eddfb79ae94b16785b696aa95aae71159dd745556e68dcbb330`, `1100fce77f322c364bb0b4038b8a697310bf29bc06f7f260716f86344981735e` |
| FP4 packed multiply Git source patch | `835d045e9eaac46ad14c52a47ca1827fb493755d4649b06fffd2a67e56ea5af6` |
| FP4 packed multiply executable and driver | `c871020b167770e2ace38380fa6f3647c10c863c737fbed0f21fb4efe51a22a8`, `52980c763aa557cef533c8be1527659a7a42c8d0f56f1a9f4c64599705070c9d` |
| FP4 packed multiply short receipt | `6a8fe69ab7d58c61aa618cd562bc0276b4563d62c199a2d7a0e57c974a7041f8` |
| FP4 packed multiply stock receipts | `cfdfa8527b18ae1b43734a30a8f91655cc9ec512d4d3c309b331b32e6a6cfe60`, `bd27974f3fc276a2c704f3805a4600b49a079fa3cc1c50609f27822d573a0dde`, `3d47a9a86bd6ea53ba30bba258989484cf655157350d0ac9c20e2c18a4f86f3c`, `4ee7e8d8a4cb21fe055df9ad4cb8d931e3bc66450808b3aa3d48239552f236e5` |
| Sampled-width combination native and worker Git source patches | `b17735290dc6022c342b3be13af8d075718e0dcf953138dc8f1814f7e1ed37a0`, `462e050108b9a82620e23a3224621cfa0f817cc6976da7235ceeb0b118e80d0a` |
| Sampled-width combination executables | `41bb998ea52e886a5fc35ffd8ef4bad2e15795a60ea28b02f8a1fdc7caa0a805`, `6d52489f6b8691a87cbf5da490fafe7e19125eff872a7c0e9168acc13ba4cb82` |
| Sampled-width combination native driver, worker scorer and worker driver | `a852790e552a7818cd185e7e5e1ee9db220834bb12e4e4356e303826fc6fccb9`, `4520498e54936c854eeabb20781e7168d586579cc1b6cf28966d63726592d5d9`, `1100818ff99bab478400dfa98ba2c19f19fbb96262d6f9f920016de36e4dc5e4` |
| Sampled-width combination native short and stock receipts | `1d3d12154694b2683a95a55694eef2ca9e5ea083e5a702f8ee77dc20be856325`, `140b64b9bd5b6038414e5de979b3aae1b7d134ab1d2352a7fcf16dc4ca72ffe5`, `9383dc394e3b6044efa7f92ddec823252b3da7ddbb1f40a7ab43673403be69bd`, `b1aee6448ce9dda91d63b5a695c1e5592ac46d5a0efac8a7dacc8e864fc04702`, `3c54d8a0eec6c86c7fdacfc68da3165cbf0301700ef33d6cbb5fcf88f6b0f0fe` |
| Sampled-width combination worker pairs and host observations | `270b9df4b273c7519b24dae136ab6fca4d2187d8e6ba091c8491479ee42c9c19`, `0b06f90c3efdca8e7dd9608c2890442554e8a4706db627ca87b3398d0db46de6` |
| Retained three-row verifier Git source patch | `a5469c8d7c94f812bb3523fe3d510abe056c011bbfc67ae2d481146793c02562` |
| Retained three-row verifier native and worker executables | `dee109cd684bfb35c61a2a1acb70ffb0fe1ef4a3a1042cb4454b6a8fb6772d95`, `e02eb6f2d1bdf3aba56f1929d79a268c33c656cdb5399bf2be994eb21491d975` |
| Retained three-row verifier qualification driver | `28d6455fc20403d2b2114df3631dd4af5530d3d88a5495adcf7e5ba17d02db78` |
| Retained three-row verifier flags-unset summary | `b8a5b540c43b3942a458316f8d3f2943ed64beadfbb40fe803a93111aac3d011` |
| Retained three-row verifier proof receipts | `b0daa01723ad01ff5649114b177b22b6b3eb9c83f57630c9843dc368ffc5f6b5`, `7295d26de74bdb28c3e80752c6ac7b085bf4b27d75b939f0d555d6ab4a509766`, `d42435a50913bc937468297faccd57399ca364d1f251287ab57950e7243c8d3d` |
| Retained three-row verifier lifecycle receipt | `d1b51cd09652bd043eb12d6546e89535af29ed286b5f587400a88200975afc6b` |
| Retained three-row verifier worker results and host observations | `a05589f7abc024b198db833a2710b73f3f08dc5c9a60cbb970726a3d90220c03`, `99988f3dfce28670d4d2f28d98686bf4dd3dabc95b5d34f66de1a84f08a82e7b` |
| Previous worker recheck driver, results and host observations | `6296687e852d8d09f7a812a372b6c42cb0642f162060de57ea37f7f76da04e5c`, `bf8ed49a3edfb16f8a1ac2cfc73d05d5cdb6906b60049756b057bb48d1366620`, `8f3a69edbf6ed9c0373071d15d919c484cc5da6200147e7700bfbe40dfca2f8c` |
| Generalized-shader recheck driver, results and host observations | `3d0d35984a704d87a9d63ae716b8c549f874d5360578e0af97adec764c713587`, `1bbbba235ba439120dcfe005efd6068630a6f710029f158a1645d2c56b745aa6`, `325c729f8bbd3d2fbc426867e1b86ce0bc0e15c30f18a566a53996944a040f6a` |
| Final three-row verifier Git source patch and qualification driver | `b6024fe0fd600d3389a2b66f01fab20cfeffb84fe2816d446a5703ee248beb8c`, `045f65f6b890d439d56ba451e15bfc9ef5a34864a530cd42e10f7a664b63ef3b` |
| Final three-row verifier native and worker executables | `85572badeaecd19a2004b55511edbe6716884eba423eb955ad2f90363f490c4a`, `eee6d6a24856464717cb12cce083242d9c835652ebadc4bdbab31f88d67bc980` |
| Final three-row verifier flags-unset summary | `b8a5b540c43b3942a458316f8d3f2943ed64beadfbb40fe803a93111aac3d011` |
| Final three-row verifier proof receipts | `5fe2f38a28b0ff2ad9d03ccf552eb6b558078a52f2993af2511a8b175b37f676`, `ea48ba9c4b141c51c9cbe60a5b43add86369fc1603858823cb29cb99ff96fbd0`, `729c17255de2be5fb2dee6a9343cab6fdba1e632afad636e9ae3386d867426c6` |
| Final three-row verifier lifecycle receipt | `d1b51cd09652bd043eb12d6546e89535af29ed286b5f587400a88200975afc6b` |
| Final three-row verifier worker results and host observations | `3124244e9dfb5634436e7a51ca3c936206cd2c9a122d73667a45526911514821`, `c2da352bf4cda5cd4d140652c6d01a0ba2c34c8eab0b044051448c197b65a072` |
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
