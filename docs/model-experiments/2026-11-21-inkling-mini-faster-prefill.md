# 20261121-INKPREFILL-04 · Faster native Inkling 2 Mini prefill

| Field | Recorded value |
| --- | --- |
| Artifact | Original vision-language checkpoint `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, target shard SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Host | 128 GiB Apple M5 Max, macOS 26.7.2 |
| Incumbent | Native 64-row Metal prefill from `12cce20d`. Release Rust 1.98.1 with locked dependencies. |
| Workload | Native 4,096-token pinned control at context 4,100. Private input bundle SHA-256 `74d6457708d31fe3ba6aeeab41b836d6ac7cd825cb3837fc3ddacc64e0c19719`. |
| Numerics | Preserve native BF16 boundaries, F32 accumulation order, causal cache writes and snapshot state. The earlier Ollama comparison has its own parity gate. |

## Attribution and design

Finer Metal timestamps split the incumbent's 32.430 s of command GPU time over 4K. NVFP4 experts took 8.869 s and the four serial MXFP8 visits took 4.348 s. Attention input preparation took 6.601 s and expert input preparation took 3.545 s. The two attention bands took 3.849 s for dense and 4.323 s for relative/cache. All 200,058 continuation logits matched the uninstrumented path. The instrumented binary SHA-256 is `e97ef6aa63b357142f481eba9b0ef699deff98ade8ed9465ecc50fb21d73c2a1` and receipt SHA-256 is `faaea5128a80a66f320183395291654fd00b7cfb50f4f945d508aa2607856450`.

The worker candidate uses 128-row commands. It shares packed MXFP8 weights across up to four routes to the same expert, with the existing per-row FP32 lane reductions and route-order sum. It computes the cross-pass gate expansions in 16-row matrix products with unchanged fragment order. Eight-row BF16 projections share weights and retain the previous four-row K lanes, dot grouping and final BF16 round. The grouped expert path covers tiles of 16 rows and up. Singleton decode keeps its old command geometry.

## Screening

Each cell uses eight alternating-order pairs over distinct windows in the same loaded model. The timer covers prompt ingestion. Comparisons use all live cache state, final BF16 taps, four singleton continuation logits and exact replay from a snapshot. Every cell won eight of eight pairs and passed all comparisons.

| Change | Window | Control | Median paired wall ratio |
| --- | ---: | --- | ---: |
| Batched cross-pass controllers | 128 | Incumbent B64 | 0.905 |
| Grouped MXFP8 | 128 | Incumbent B64 | 0.927 |
| 128-row expert and command tiles | 128 | Incumbent B64 | 0.918 |
| Batch exact relative/cache attention | 128 | Incumbent B64 | 0.985 |
| Wider tiles, controllers, MXFP8 and batched attention | 128 | Incumbent B64 | 0.733 |
| Add batched attention to wider tiles, controllers and MXFP8 | 128 | Same path with per-query attention | 0.979 |
| Add eight-row shared BF16 projections to wider tiles, controllers and MXFP8 | 128 | Same path with four-row projections | 0.962 |
| Group smaller tails | 31 | Same B128 path with a 64-row minimum | 0.395 |

The exact batched-attention trial staged only the cache slots due to be overwritten, then shared dispatches across independent 16-query groups while preserving each query's reduction. Its first full 4K combined trial took 23.424 s against 32.059 s and preserved the full native state and continuation. The separate marginal gain was about 2% for 13.4 MiB of extra workspace, so the production candidate uses per-query attention. The eight-row BF16 projection is part of the combined candidate and uses the same workspace as the four-row variant. Grouping 31 rows removed about 343 ms from the median uninstrumented tile, from 567 to 224 ms.

The two screening binaries have SHA-256 `107293d394a91cf5552edfc37bcacc43c8213a3c10fc3f0035412d24bb74c3ab` and `faa096cbfa14aec9ded2e04f29bd438aa279b477c8084bfd3e8e4426c9db9adc`. The first full 4K aggregate SHA-256 is `3c603c5f03bef4774b1b027f9d03dbc6f78f324fcc062a07154c34a5b6edfcbb`. The private marginal-screen patch SHA-256 is `0172c3ad28a91897c237f81c5e35c41ce31bc780acf5af04dbfffa5f2e165a51`. The first integration build caught a stale tuple type in the controller's projection call. Updating the binding to the new offset-capable signature resolved it before performance collection.

## Final model control

In eight matched 159-token windows spanning partial tiles, the production configuration won every pair. Its median paired wall ratio was 0.584, from 0.580 to 0.589. A full 4,096-token run took **23.281 s** against **32.029 s** in one same-session pair. All state, target taps, four continuation logits and snapshot replay checks passed in every pair. The additional Metal allocation was **94,191,616 bytes** (89.8 MiB) for the candidate, within the worker's existing 128 MiB workspace reserve. The same short and full checks can be run from a private fixture.

```sh
inkling_slice --prefill-speed "$ORIGINAL_SHARD" "$FULL_ORACLE_4096" 159 wide+controllers+mx+small+widegemv
inkling_slice --prefill-speed "$ORIGINAL_SHARD" "$FULL_ORACLE_4096" 4096 wide+controllers+mx+small+widegemv
```

Final example SHA-256 is `dd881e2655e726675e79adb715427e377837a71ae6ea1b0bbef53689ebe1f31e`. The tail and 4K aggregates are `333eef23073e949431e41789c97c35f747cc8ab5e943f3e91915131ec4e54fd5` and `a7c68ba80e801b699fd38ce2ebb867565befb89f759caf2961d18c0281f1c45f`.

The production worker reached 174.95 input tokens/s greedy and 175.96 sampled with the original vision-language checkpoint, an increase of about 40–41% in each matched pair. Both output hashes matched, and decode cleared its existing budgets. The worker's prefix boundaries remain 1,024 tokens, aligned to eight 128-row commands. Cancellation is checked at each command boundary. [Worker benchmark](../benchmarks/2026-11-21-inkling-mini-native-worker.md#128-row-worker-prefill)

The 1,031-token worker control reused 1,024 tokens and reproduced all 24 output tokens. Prefill took 5.714 s cold and 0.144 s with the snapshot restored. After cancellation, the next request reused zero tokens and reproduced the same output. The local desktop backend streamed text, passed the image color control and completed a tool-result round trip. Draft-on and draft-off responses matched in the greedy and seeded sampled checks with vision resident. The prefix and desktop aggregates are `8c2a925f8fbbc4636f59aff20698c1998fb41de4214f99fe8d7d82ba7ab513fb` and `c5b6bd59db828f2dccedc7f61a9cde84763ac40da6ec66cb6d9e25d1f46ebf26`.

The short and 4K pinned slice aggregates reproduced both prior hashes exactly. Locked `stark-generate` tests passed 157 cases, with five private-artifact tests ignored. Rust formatting and repository policy checks passed.
