# Inkling 2 Mini decode toward 80 tok/s

These experiments use the original Inkling 2 Mini target and DSpark checkpoint on the 128 GiB Apple M5 Max, with the combined short verifier and batched Q, K and V controllers enabled. The build starts at `90fce867` and uses locked Rust 1.98.1. The [integrated controller record](2026-11-21-inkling-mini-decode-60.md#20261122-inkdecode-78--integrate-batched-q-k-and-v-controllers) binds the checkpoint hashes, selectors, parity and earlier paired baseline. The target budget is 12.5 ms per useful output.

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
