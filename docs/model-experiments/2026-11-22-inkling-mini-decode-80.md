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
