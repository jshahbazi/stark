# 20261121-INKPROFILE-04 · Default exact one-row router for Inkling 2 Mini

## Promotion scope

The original Inkling 2 Mini worker now selects the vector-loaded one-row router within M5 Max source text mode. The gate covers text prompts of at least 512 tokens, context at most 4,352, drafting off and the existing memory admission. The singleton kernel can run on the prompt tail, generation seed and subsequent decode. Bulk prefill continues to use the packed router. [Worker admission](../../crates/stark-generate/src/inkling.rs) · [Default prefill qualification](2026-11-21-inkling-mini-prefill-default.md)

The router retains the source `16 × 32 × 16` tensor operation, the distinct 2,048- and 256-element accumulations and the existing expert selection and normalization. The worker selects the new entry after its Metal pipeline registers. If registration fails, it can compile the ordinary source library with the one-row shader excluded and use the original source router. The same snapshot layout applies to both routers. [Opt-in investigation and exactness controls](2026-11-21-inkling-mini-nvfp4-router.md)

## Artifact and run controls

| Field | Value |
| --- | --- |
| Original checkpoint | `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` |
| Target shard SHA-256 | `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Resident DSpark shard SHA-256 | `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2` |
| Reference runtime | Ollama `6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8`, stock-schedule capture |
| Control worker | Stark `06d8a96575713a7880c098d2b0d4ae706f40bbed`, original source router |
| Candidate worker | Default-router patch atop `06d8a965`, same promoted prefill kernels |
| Environment | 128 GiB Apple M5 Max, macOS 26.7.2 build 25G332, Metal 32023.883, locked Rust 1.98.1 release builds |
| Worker timing | All `STARK_INKLING_*` variables unset in both arms, eight alternating pairs per mode, 4,096 prompt tokens at context 4,352, 256 emitted and sampled tokens, zero reuse, vision and DSpark resident, drafting off |
| Component timing | Four pinned stock 4K states. Restore the same state for both singleton routers and alternate order in instrumented and plain commands. |

The acceptance checks require pinned short and 4K parity, selection of the new kernel with flags unset, exact router outputs at all visits and equal generated hashes. The user requested default adoption of the exact kernel. This study measures the useful-decode effect against the prior router on the current prefill policy.

## Flags-unset parity and one-token commands

The connected short and 4K slice aggregates matched the qualified record. The four-token full target had zero unequal pinned F32 elements. The four stock 4K cases matched **16 F32 logit vectors**, **768 BF16 visit residual blocks**, and four snapshot replays. New and prior builds matched the native cache, F32 readouts and draft taps on all five full-target fixtures. The new default also matched the prior build with its router flag explicitly enabled on the four-token input and one stock 4K input.

The stock profiler selected `ink_singleton_router_source`, retained both default fast prefill kernels, and reproduced the first pinned continuation. At the same restored 4K cache, the new router matched all **128 logits, eight selected IDs and eight expert weights** per visit at all 48 visits on each of the four inputs. The ordinary source shader compiled separately with the optional entry disabled. The 512-row source-disable control matched the prior fallback's cache, logits, residuals and draft taps and replayed its snapshot exactly.

| One-token boundary | Original router median | One-row router median | Median paired saving | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Instrumented router projection over the 44 NVFP4 visits | 3.473 ms | 2.527 ms | +0.884 ms | 8/8 |
| Plain whole command | 34.432 ms | 33.610 ms | +0.863 ms | 8/8 |

## Worker results

Eight pairs per mode passed all fixed-work controls and matched their output hashes. Useful decode counts the 255 intervals after the first emitted token. The comparison is the median of the paired rate ratios.

| Sampling | Original router | Default one-row router | Paired change | Candidate range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 32.57 tok/s | **33.78 tok/s** | +3.15% | 32.02–34.14 | 8/8 |
| Seeded sampled | 31.29 tok/s | **32.12 tok/s** | +2.93% | 31.22–32.30 | 8/8 |

Median paired decode latency savings were +0.941 ms per interval greedy and +0.918 ms seeded sampled. The prefill controls moved +0.69% and +0.39% at paired medians. The default's prefill medians were 1,023.44 tok/s greedy and 1,013.32 tok/s seeded sampled. The measured useful-decode effect is below the standard 10% default-optimization threshold. This promotion follows the explicit request for the previously qualified exact router.

The workers accounted for 91.10% of GPU counter increments over 218 measured intervals, including idle gaps. Within estimated decode windows, worker shares were 94.58% for the control and 94.73% for the candidate. Host observations covered nominal and fair thermal states. Pageout and swapout counts changed by +55 and +0 pages of 16 KiB. The scored windows ran without builds or other benchmark work.

The flags-unset worker restored a 2,048-token prefix, matched the cold changed-tail output, rejected both invalid saved boundaries, and cleared saved state on cancellation. It matched the previous worker on packed tails of 16, 47 and 64 rows. Both worker readiness checks verified vision and DSpark capability.

## Evidence identities

| Artifact or private receipt | SHA-256 |
| --- | --- |
| Locked dependencies | `07b6d42497825a651719685a28a107bfc0a32119b8e8b8adeab2d6c01a96695b` |
| Prior worker | `3bd502cb7f7c86b03355f69ab3f2cbe93707b30d85b81f597ba9f2c4b6a68c88` |
| New worker | `b6f1ac0ff931b2440dbb76620ad98d4b26efa8f9edd0cfe367876edef41e2abe` |
| New native example | `45cb9916017f4c86a0f5f1dea48d8443d715922657e5da8fd159d378cb7f238c` |
| Default selector and same-state router profile | `2bf556cdba5fb46f39db04cd95161f55c0440af387f6483a64bfbf283366dca8` |
| Flags-unset paired harness | `5e3509a128b2e81838f31720ae2a7b91a94f9158ca4d9f2445c4cb9f41bb2670` |
| Worker pairs | `49041281a20b231c64303ce2235ea7d7419d01c4f9aa45e2dfefe2ae5c597890` |
| Host observations | `544b50ebc6cd3bdfe1226943f0d2ee7b7820f2d36e606b372a5fdbbb6018c6fc` |
| Prefix and cancellation controls | `8d02db1c90a2134e4bf1b50b052f832ed03748fb2c9b5828d728452758c74f10` |
| Short packed tails | `f431d8684afff1117d665c127767293d734d8ef35db4b5c4763f908ae5b81da5` |
| Source-disable comparison | `c897196c2ad9918d934a764ade96c05850e2ebb70a2734a012045752df3bd1ff` |
| Default-router source patch against `06d8a965` | `555b3c072fe44bbfec47dd07cef151d80a5ae346948c03865a01ab69708a8f5f` |
| Prior explicit opt-in short replay | `6ab3fc6c4162d6fb4ab6cecbf267cc35e1bb48c98cb1ca5a1485964a9052af16` |
| Prior explicit opt-in stock 4K replay | `38a5269b4754a4a44e566687ff26b4b6020730ab6a9e1997ef0fa7c3456fdd90` |
