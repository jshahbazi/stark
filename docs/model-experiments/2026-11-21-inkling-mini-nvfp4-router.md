# 20261121-INKPROFILE-03 · Inkling 2 Mini NVFP4 guard and singleton router

| Field | Value |
| --- | --- |
| Attempt | `20261121-INKPROFILE-03`, 2026-11-21, Codex |
| Artifact | Original fixed-three-pass Inkling 2 Mini revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. Target SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. Draft SHA-256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`. |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332). |
| Predecessor | [Stock prefill and singleton attribution](2026-11-21-inkling-mini-stock-profile.md) |
| Build | Locked Rust 1.98.1 release on `60a8efb823bfb60b80330b2e9655f44c1317edd7` plus the experiment changes. |

## Real-route occupancy and exactness replay

Two pinned stock prompts exercised both bulk widths, 2,048 and 2,044. Each bulk supplied 44 NVFP4 gate/up and 44 down products. The instrumented exact dispatch counted live tiles, routed rows, the broad guard failures and per-output replay entries. Each shadow timing repeated gate/up and down on three sampled visits in each bulk with identical inputs, weights, route runs and kernel geometry. The last exact dispatch overwrote all shadow products before the epilogue. Both full prompts reproduced all pinned seed and next-token logits.

| Metric | Gate/up | Down |
| --- | ---: | ---: |
| Live row capacity filled | 88.85–89.05% | 88.85–89.05% |
| Launched run slots with live rows | 89.95–90.18% | 89.95–90.18% |
| Active SIMD K-blocks failing the broad guard | 8.01–9.13% | 82.35–89.00% |
| Output K-block entries replayed | 0.00062–0.00077% | 0.278–0.332% |

Each timed set covered six matrix products. The exact arm used 73.66–92.08 ms, the guard-only arm 71.79–88.51 ms, and the tensor-product arm with both guard and replay off 46.77–56.55 ms. Removing only the replay loop saved **2.5–4.0%** of exact shadow time. Bypassing the guard as well accounted for another **33–35%** of exact shadow time. These are attribution controls. The guard-only and exact runs both write telemetry, while the bare arm avoids guard-related counter updates. Their timing differences bound the cost in this instrumented comparison. Only the ordinary exact result contributes to model execution.

```sh
cargo +1.98.1 run --locked --release -p stark-engine --no-default-features \
    --features inkling-slice --example inkling_slice -- \
    --probe-stock-nvfp4 "$ORIGINAL_SHARD" "$DRAFT_SHARD" "$STOCK_ORACLE"
```

## Candidates

- **Shared scales.** Each weight-loading thread decodes one 16-value scale block and deposits its peak and binary quantum in threadgroup memory. Both SIMD row halves reuse those values. The existing guard comparisons, K order and replay remain exact. The source-scoped experiment selector is `STARK_INKLING_EXPERIMENT_NVFP4_SHARED_SCALES=1`.
- **One-row router.** Keep the source `16 × 32 × 16` tensor operation and its distinct 2,048- and 256-element accumulations. Load only the live activation fragments and vector-load the aligned BF16 weights. The source-scoped experiment selector is `STARK_INKLING_EXPERIMENT_SINGLETON_ROUTER=1`. The narrower `16 × 16 × 16` screen failed Metal compilation. The SDK requires at least one dimension of 32 when both inputs are cooperative tensors in this path. That variant yielded no timing and was removed.

In the two-fixture singleton screen, every checked logit, expert ID and expert weight matched the original projection at all 48 visits. Across four instrumented pairs, the 44-visit NVFP4 router projection was 2.34–2.50 ms in the candidate versus 3.19–3.56 ms in the original. Median paired saving was **0.92 ms**. Uninstrumented single-token commands varied by order, with a median paired saving of **0.31 ms** in this short screen. Use the worker comparison for the throughput decision. Both isolated candidate builds matched the pinned first continuation on both stock prompts.

## Parity after both changes

The archived connected short and 4K slice checks produced identical parity aggregates. A separate combined full-model run after a four-token prefix reproduced all four pinned F32 output vectors. Baseline and candidate matched native prefill cache, visit residual reports and draft taps, and both replayed the snapshot exactly. Four stock 4K captures matched all **16 F32 logit vectors**, all **768 BF16 visit residual blocks** and all four snapshot replays. The target diagnostic forced source arithmetic for the short full-model check. Normal short worker requests retain their existing source-mode admission rule. With both candidates enabled, the worker restored a 2,048-token prefix into a changed 471-token suffix, reproduced all 24 output tokens, rejected the invalid 2,048- and 4,096-row boundaries, and cleared the saved prefix after cancellation.

## Worker comparison protocol

Use one sealed release binary and run three resident workers with the same original checkpoint. Enable neither candidate for the baseline, enable shared scales for one arm and the one-row router for the other. Alternate request order across greedy and seeded sampled windows. Require exactly 4,096 input tokens, context 4,352, zero reused inputs and exactly 256 emitted and sampled tokens with vision and DSpark resident and drafting off. Each eligible window requires all output hashes to agree and to reproduce the archived hash where available. Use four eligible windows per mode for the screen. Keep the original defaults unless a later promotion run establishes a practical gain within the playbook's regression and lifecycle gates.

## M5 Max worker results

Four eligible matched windows per sampling mode produced 24 timed requests. All fixed input, output and reuse counts passed, and the three arms matched output hashes on every window. The initial launch failed the front-end image checks with the native template helper absent from the harness path. The successful session used the existing helper, completed fresh image checks for the sealed binary and reused that identity-bound receipt across its other two arms.

| Primary metric | Stock median | Candidate median | Paired median change | Candidate wins |
| --- | ---: | ---: | ---: | ---: |
| Shared scales, greedy prefill | 802.14 tok/s | 827.13 tok/s | **+1.61%** | 3/4 |
| Shared scales, seeded sampled prefill | 753.16 tok/s | 800.87 tok/s | **+1.77%** | 4/4 |
| One-row router, greedy useful decode | 32.88 tok/s | 33.34 tok/s | **+1.64%** | 4/4 |
| One-row router, seeded sampled useful decode | 31.21 tok/s | 32.01 tok/s | **+2.52%** | 4/4 |

Paired ratios and arm medians summarize different quantities. In particular, the sampled prefill arm medians reflect spread across the four windows. With shared scales, median paired useful decode moved −0.74% greedy and −0.04% sampled. Request wall time moved +0.14% and −0.79%. With the one-row router, median paired prefill rate moved −1.26% and +0.83%. Useful decode time fell by a median of **0.49 ms per interval greedy** and **0.79 ms sampled**. Whole request time moved −0.04% and −1.81%.

The three workers accounted for 99.81% of observed Metal counter increments over 176 covered intervals. Host samples included nominal and fair thermal states, with no before/after thermal or performance warning. System pageouts rose by 124 pages of 16 KiB, and swapouts held steady. All eligible pairs are included. This was a small screening run. The scale change is close to timing variation, and the router recovered part of the original 1.8 ms worker gap. Both remain opt-in for research. The stock path remains the default on the branch.

## Evidence identities

The sealed worker SHA-256 is `da4b53b5e97ff80f1c5da5e77dc2b620174969e05315c5ae3ffb54bf51b1e9d5`. The parity executable SHA-256 is `efa6d3bbaff7ef19333a4a221cfd8e3fd48cc621493d1412b87a233cbfd18a93`. The corresponding measured engine diff is `ff6e39b9c5fae180422ef90db56fa7f984a1ac82fa1aacfd0e6cb8ff7ecee31c`. The final patch also identifies the telemetry kernel explicitly in the probe's JSON report.

| Receipt | SHA-256 |
| --- | --- |
| NVFP4 occupancy and shadow timing | `6a5fdb76fa464c8805cc6e872251c3bc2627ee39992466009b59a07f111e792f` |
| Initial exact router screen | `4a604a08c98d5594ef285b7c6af84929d7f503e837d20c66c924aa949e96c02b` |
| Initial shared scales screen | `54bbbbe5ba8d155d5ce6432f3ce03374cfcdfba8508ae217dc9e126cc2ff2bfb` |
| Matched worker screen | `84a6039e0007cd9d5b1766a3b830974087fd5a4156b18cd48fe9641d1c00d085` |
| Worker host aggregate | `ad670f9d0e9ce46942a6e0acc6d315d6d78efee42d05f28698ce60dc05dae7a4` |
| Combined worker lifecycle | `120a10148c5bc5a300965b72a642c4aba632dd1cf8cc8de23b1c7d0588a2ba73` |
| Short and 4K parity summary | `d64c665273dfddc248584f40accf0d3723d776ac581e46cb3e8a942f4e626ec9` |

Private aggregate receipts and the sealed binaries stay in the ignored `.inkling-local/nvfp4-router` directory. [Stock oracle capture protocol](../archive/experiments/2026-11-21-inkling-oracle/README.md#stock-schedule-4k-text-capture)
