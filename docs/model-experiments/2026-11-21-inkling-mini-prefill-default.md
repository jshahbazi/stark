# 20261121-INKPREFILL-14 · Default fast prefill for Inkling 2 Mini

## Promotion scope

The native worker selects 64 × 32 NVFP4 tiles and compensated K32 MXFP8 with masked source-order replay for the original Inkling 2 Mini checkpoint on M5 Max. The source prefill admission continues to require the publisher artifact, sufficient headroom, text input of at least 512 tokens, context at most 4,352 and drafting off. Bulk commands above 64 rows can use the tensor path. The worker retains the existing visit cache layout and its 2,048-token snapshot boundaries. [Original checkpoint admission](../../crates/stark-generate/src/inkling_artifact.rs) · [Opt-in kernel qualification](2026-11-21-inkling-mini-mxfp8-tensor.md)

The worker builds the ordinary source path first. The NVFP4 selection commits after both its tile and bucket pipelines are available. MXFP8 publishes the tensor workspace and its per-bank norms after every bank has initialized. Setup errors leave the corresponding ordinary source kernel available. The 1,344 MiB prefill reservation covers the faster workspace. `STARK_INKLING_SOURCE_PREFILL=0` selects the smaller 512-row workspace at startup.

The promotion keeps the measured arithmetic. The BF16 certificate still uses the M5 tensor accumulator assumption exercised in the earlier direct audit of 3.78 billion accepted results. The audit evidence covers this checkpoint and the measured M5 Max software stack. [Numerical bound and audit](2026-11-21-inkling-mini-mxfp8-tensor.md#20261121-inkprefill-12--k32-compensation-and-vector-replay)

## Artifact and run controls

| Field | Value |
| --- | --- |
| Original checkpoint | `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` |
| Target shard SHA-256 | `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Resident DSpark shard SHA-256 | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |
| Reference runtime | Ollama `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`, stock-schedule capture |
| Prior worker | Stark `6e141676ea3ea7e8af76b324fc23f01c3afd7046`, both qualified kernel flags explicitly enabled |
| Candidate worker | Default-selection patch atop `6e141676`, all `STARK_INKLING_*` variables unset |
| Environment | 128 GiB Apple M5 Max, macOS 26.7.2 build 25G332, Metal 32023.883, Rust 1.98.1, locked release builds |
| Measurement | Eight alternating pairs per sampling mode, 4,096 input tokens at context 4,352, 256 emitted and sampled tokens, zero input reuse, vision and DSpark resident, drafting off |
| Acceptance | Pinned short and stock 4K parity, both intended kernels confirmed with flags unset, matching output hashes, and a cold prefill median of at least 1,000 tok/s in both sampling modes |

The benchmark measures the final flags-unset worker against the previously qualified kernels in the same session. It tests whether default selection preserves the speedup. The prior study tested the speedup against ordinary MXFP8 with the same wide NVFP4 tiles.

## Flags-unset parity

The new example reproduced the archived connected short and 4K slice aggregates. The four-token full target and four stock 4K captures matched the prior opt-in build's native cache hashes, continuation logits, per-visit BF16 residuals and draft taps. The stock 4K captures reproduced 16 pinned F32 logit vectors and 768 BF16 visit residual blocks, with exact snapshot replay in all four cases. The short capture had zero unequal F32 elements and replayed its snapshot exactly.

The flags-unset stock diagnostic reported `ink_group_qmm_reused_guard_rows64` and enabled MXFP8 tensor replay on the two bulk commands. Its source-mode continuation matched the pinned logits and the diagnostic audit mode was inactive. The full-target allocation delta was 1,259.72 MiB in each flags-unset parity control, matching the prior opt-in workspace.

The explicit source-disable test compared both release builds on one 4K input. Both selected 512 rows and matched the prior fallback's native cache, continuation, visit residual and draft-tap hashes. Both replayed the snapshot exactly.

## Worker results

All 16 pairs matched output hashes and passed the fixed-work controls. Rates are arm medians. The change is the median of the paired rate ratios.

| Sampling | Prior opt-in | Flags unset | Paired change | Flags-unset range | Requests at or above 1,000 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 1,030.58 tok/s | **1,019.54 tok/s** | -0.22% | 939.66–1,052.06 | 6/8 |
| Seeded sampled | 1,014.84 tok/s | **1,018.31 tok/s** | -0.19% | 990.07–1,055.74 | 7/8 |

The new default crossed the 1,000 tok/s median threshold in both modes. Median prefill took 4.018 s greedy and 4.022 s seeded sampled. Useful decode medians were 32.80 → 32.78 tok/s and 31.26 → 31.28 tok/s.

The workers accounted for 98.05% of GPU counter increments across 219 measured intervals. The host observations covered nominal and fair thermal states. Pageout and swapout counts changed by +87 and +0 pages of 16 KiB, respectively. The scored intervals ran without builds or other benchmark work.

The flags-unset worker matched cold and restored-prefix output with 2,048 tokens reused, rejected both invalid packed boundaries, and cleared its saved state on cancellation. Three short packed tails at 16, 47 and 64 rows matched the prior opt-in worker. Worker readiness verified the image path and DSpark capability. This record qualifies the rebuilt worker on the named checkpoint with drafting off during timed requests.

## Evidence identities

| Artifact or private receipt | SHA-256 |
| --- | --- |
| Locked dependencies | `07b6d42497825a651719685a28a107bfc0a32119b8e8b8adeab2d6c01a96695b` |
| Prior opt-in worker | `bc5de5f5336093eb27c82ade44b9422f720f4a4035c08a8e498ba05d1f407fb9` |
| New worker | `3bd502cb7f7c86b03355f69ab3f2cbe93707b30d85b81f597ba9f2c4b6a68c88` |
| New native example | `6ee5f5019b2cd33abaf5e8cff54e62f3894e49d80cd3135fad2e0c42bbffdf49` |
| Default selector profile | `9e851fc2d92d6b3c99fd6d894abeefc129407880e94672ad3fdef2900cea31fa` |
| Flags-unset paired harness | `02a0df0f3ef6ca61f10439e6c940e93c2bb4e0df45aaeb69873c65cf5bca6470` |
| Worker pairs | `bed1520a3d182d310ea01ef721e5946cd8bb9af0c81c2bca7ffceeb0fdbe83ab` |
| Host observations | `608b07324678f9aad130d72725bb1e90a8577e407d2f4b350621b9c1369c8a47` |
| Prefix and cancellation controls | `25676fdfb8bde9703132b19a4b5d96a2c42305b6620eb7958c38c8bb65b7fdae` |
| Short packed-tail controls | `f431d8684afff1117d665c127767293d734d8ef35db4b5c4763f908ae5b81da5` |
| Prior source-disable comparison | `c19f36c79d25aeebc5b843329e326b90cc2e758306c6da83c298994894173e07` |
| New source-disable comparison | `69f3a77cad96b36ecf15f5a01af6e00fad5150a70ef4c0a185f9ac18baad2987` |
| Default-selection source patch against `6e141676` | `2cbe0349e3b46c67b960a44a942783700546ebb05954568716897be07e160b15` |
