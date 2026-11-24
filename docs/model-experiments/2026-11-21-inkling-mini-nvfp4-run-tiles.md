# 20261121-INKPREFILL-10 · Inkling 2 Mini wider NVFP4 run tiles

## Scope and target

This study targets **1,000 input tok/s**, or at most **4.096 s** for a cache-cold 4,096-token worker prefill on the 128 GiB M5 Max. The original fixed-three-pass checkpoint is pinned at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. Target and DSpark SHA-256s are `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` and `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`. The stock scheduler remains `2048 + 2044 + 3 + seed`, context 4,352, vision and DSpark resident, drafting off. The previous controlled screen measured 802.14 greedy and 753.16 seeded sampled tok/s at the stock arm medians, so the observed gap was about 1.02–1.34 s. [Previous worker screen](2026-11-21-inkling-mini-nvfp4-router.md#m5-max-worker-results)

## Exact kernels and screening results

The NVFP4 guard already checks a conservative bound and replays the entries that cannot be certified. The M5 Max destination-fragment probe confirmed that each lane owns eight columns across two rows of its `16 × 32` result. The candidate gathers the two row bounds and eight column bounds once per broad guard failure. It preserves the existing floating-point bound, per-entry quantum, replay loop and tensor K order. The larger workgroup tiles share the decoded weight columns across four or eight SIMD row fragments within one workgroup. They leave the packed weight format and the 32-wide K step in place. The 64-row run table fits in the existing allocation sized for 32-row runs.

The MXFP8 source-wide path called the four-route dot twice. A fused variant shared decoded weights and scales across the eight routes, leaving the order of each route's products and scale accumulation intact. The first combined run and an isolated compiler probe found that a zero register index literal matched both Metal accessor overloads. Giving the index a typed variable fixed the compilation. No timings came from those failed launches.

Each screen used two pinned stock inputs and one loaded native target. Both candidate and control ran the same rows in separate timestamped and uninstrumented passes. Order alternated within each timing stratum across inputs. All instrumented and uninstrumented arms reproduced the pinned seed and first continuation logits.

| Candidate | Uninstrumented stock → candidate wall, two 4K inputs | Instrumented NVFP4 matrix difference | Decision |
| --- | --- | --- | --- |
| Reuse guard operands in 32 × 64 tiles, fuse both MXFP8 sides | 4.654 → 4.883 s, 6.101 → 5.496 s | −77 ms, +45 ms | Whole-prefill timing moved in both directions. MXFP8 down was 7.3–12.2% slower. |
| 64 × 32 NVFP4, fused MXFP8 up only | 4.737 → 4.898 s, 5.787 → 5.306 s | −381 ms, −337 ms | Larger NVFP4 tile explained the gain. MXFP8 up was 1.6–6.9% slower here. |
| 64 × 32 NVFP4 alone | 4.668 → 4.686 s, 5.504 → 5.216 s | **−380 ms, −319 ms** | Selected for worker timing. Gate/up fell 15.4–17.3% and down fell 8.9–13.2%. |
| 64 × 64 NVFP4 alone | 4.957 → 5.042 s, 5.547 → 5.304 s | −318 ms, −349 ms | Slower instrumented target and lower best uninstrumented rate than 64 × 32 in these screens. |
| 128 × 32 NVFP4 alone | 5.015 → 5.547 s, 5.556 → 5.382 s | −186 ms, −130 ms | Smaller matrix savings and an instrumented whole-target tie. |

Only the selected 64 × 32 experiment remains in the worker code. The opt-in selector is `STARK_INKLING_EXPERIMENT_NVFP4_ROWS64=1` and applies within the source-mode grouped NVFP4 path. It does not extend existing source-mode hardware or context eligibility. The diagnostic `--profile-stock-4k` compares that selection directly against stock using the same loaded target. The rejected variants and raw aggregates stay in the ignored experiment directory.

## Exactness

The sealed candidate reproduced the archived connected short and 4K parity aggregates. The four-token full-model comparison reproduced each pinned F32 vector, the native prefill cache, draft taps and snapshot replay. The four stock 4K inputs reproduced **16 F32 logit vectors**, **768 BF16 visit residual blocks** and all four snapshot replays. These checks include the generation seed and the first continuation rows. The worker comparison checks generated output separately. The worker also matched all 24 generated tokens on each of the 16-, 47- and 64-row packed tails. The lifecycle control restored a 2,048-token prefix, matched the changed tail, rejected both invalid packed boundaries and cleared the saved state on cancellation. Running the rebuilt telemetry probe with the 64-row flag set retained stock occupancy geometry and exact pinned logits.

## Worker protocol

Benchmark one sealed binary in two sequential request arms with only the NVFP4 tile selector changed. Alternate greedy and seeded sampled launch orders across four eligible windows per mode. Require identical output hashes with 4,096 input tokens, 256 emitted and sampled tokens, no input reuse and context 4,352. Preserve the original defaults during screening. A favorable component split or one fast request is weaker than a sustained whole-worker median.

## Worker results and target gap

The sealed worker used locked Rust 1.98.1. All 16 requests passed the length and output-hash controls. Both arms used the original checkpoint with vision and DSpark resident and drafting off. Values below are arm medians. The gain uses the median of the four paired rate ratios.

| Sampling | Stock prefill | 64 × 32 prefill | Paired gain | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 756.05 tok/s | **800.28 tok/s** | **5.84%** | 4/4 |
| Seeded sampled | 688.82 tok/s | **763.88 tok/s** | **10.24%** | 4/4 |

The candidate's prefill medians were 5.129 s greedy and 5.375 s sampled. The best request reached 868.77 tok/s. Median paired request wall time fell by 1.62% and 3.74%. Useful decode rates moved −0.87% and −0.09% at the paired median. The 64 × 32 route remains opt-in after the short worker screen. At these timings the 1,000 tok/s goal still requires removing **1.03–1.28 s** from the complete 4K prefill.

The workers accounted for 99.81% of sampled Metal counter increments across 123 intervals. Host samples covered nominal and fair thermal states. The macOS snapshots had no thermal or performance warnings. System pageouts rose by 346 pages of 16 KiB, and swapouts were unchanged. The paired comparison is the measure of gain under these conditions.

The MXFP8 expert stage is still about 1.4 s in the retained geometry's attached profiles. Early matrix screens showed a large matrix-throughput opportunity and changed arithmetic. The later guarded implementation recomputed weight norms on each tile and replayed exceptions inline. A new screen should time the matrix, BF16 certification and compacted source-order replay separately on live stock routes. Move invariant weight norms out of the tile. Set the error envelope from the current source's per-lane recurrence and the measured tensor accumulation depth. If its replay share and total cost fit the remaining budget, proceed to the pinned BF16 boundaries and 4K worker tests. [Earlier matrix and guard screens](2026-11-21-inkling-mini-packed-prefill.md#kernel-screens)

## Evidence identities

| Artifact or receipt | SHA-256 |
| --- | --- |
| Initial fusion and guard example | `90088138bd4ddf074792c78aa2c9ca91b46a31d022cfc65fe35e841e3434d6dd` |
| 64 × 32 and up-only fusion example | `362c02eef2f3d70bb14f4a77e46f9fe4f4d51a19f92a84f370e25bca22c3df5f` |
| Geometry comparison example | `55f22a0f6be12ab7be64d428bcb0f84dea2fb811d4e25df7b071f55f8e95604f` |
| Archived screen patch | `a7238b5a2abfd5b9dd26898d1d8e92380d4473edf61ea909f87831c960e249d8` |
| Lane-layout probe | `a433d544b15fe2c5a93a08c1e1085c18246d585a47df1355e9972458bb1fbb8c` |
| Guard reuse and both fusions | `8a2bb28b4844e8c60afe7b22d4d8300d767859eca18cd557f3bf28fe477a9814` |
| 64 × 32 plus up fusion | `03a9ebfcc9251f749ff8718e30a35c6559a2961dfb53e64eb4bd6be33edc2eb5` |
| 64 × 32 alone | `48557d1c9b737a6f1abb93fe28dae7c33e6e913f1755f19c461994a7a8efcf9d` |
| 64 × 64 alone | `72f7202e13446cde75beb1f93bdcaac663de4f0db366f2181eaf960db7b9fbba` |
| 128 × 32 alone | `1c1f19a8dc2d67b844d968e1b4deb8375d7583a64bf3bc9a0d4a3426de49c96f` |
| Matched worker and parity engine patch on `e3bf8275` | `7f1e0be6095931a804bb60a44598096d32b6c466f41fafc6425a9f17470638f0` |
| Sealed worker | `330e8600703811ce7730efdf5541f1fa0c3f834b85ec970e65150c236699404d` |
| Parity example | `fc5432421fee336809d764823b22f15617e3590853b344521cca2873c61e3913` |
| Connected and full-target parity summary | `d64c665273dfddc248584f40accf0d3723d776ac581e46cb3e8a942f4e626ec9` |
| Packed-tail outputs | `f431d8684afff1117d665c127767293d734d8ef35db4b5c4763f908ae5b81da5` |
| Prefix and cancellation | `744d718d76694ee4f0888242a31b7da31da60ed2b951a92d8e2bda5eb8c86a5b` |
| Final probe with the 64-row flag | `847f39165ae23e50b6e6264edfe3de8f30bb3bb927c9c9740207d77fab194319` |
| Worker screen | `ec613b78f96029d95738e1d14f2e912dafe411ec22f4b538a35876e54acb7f0a` |
| Worker host aggregate | `a2616259d6de0510a4bc8894c8666973b7de0573a5d16746a8e71bde8971ef8f` |

All receipts stay private. The final source adds a defensive guard that keeps the original QMM geometry when the telemetry probe has exhausted its dispatch slots. The worker and parity timings use the sealed build identified above.
