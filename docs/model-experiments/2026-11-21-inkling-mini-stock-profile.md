# 20261121-INKPROFILE-02 · Inkling 2 Mini stock 4K prefill and singleton decode

## Identity and measurement

| Field | Value |
| --- | --- |
| Attempt | `20261121-INKPROFILE-02`, 2026-11-21 |
| Artifact | Original fixed-three-pass Inkling 2 Mini at revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`. Target SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. Draft SHA-256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`. |
| Build | `442cd72db462e0def41784ef004fbbec241be652` plus the profiling patch. Locked Rust 1.98.1 release, `--no-default-features --features inkling-slice`. Executable SHA-256 `251c808cbbe6b479cc1d993cdc848638dafdac2adc269a2e8156514d60446fb8`. Engine diff SHA-256 `e5de1774c96ab9ecca6209008031a556ff8f14c09d183e476b462c3ec2bffb52`. |
| Host | Apple M5 Max, 128 GiB, macOS 26.7.2 (25G332), AC, power mode 2. Nominal thermal state on 25 host samples. Zero system pageout and swapout deltas, no thermal or performance warning. The profiler accounted for 99.74% of observed Metal counter increments across 23 covering intervals. An unrelated Cargo process appeared in one host spot check. |
| Prefill | Four pinned Ollama stock-schedule inputs, two each from the greedy and seeded sampled worker fixtures. Cache-cold, 4,096 inputs, context 4,352, vision and DSpark resident, drafting off. Same loaded native target, one ordinary and one instrumented replay per input, alternating replay order. |
| Decode | First teacher-forced token at position 4,096, after prefill and the prompt seed. Reuse the same restored stock cache for source and legacy singleton arithmetic. Two uninstrumented and two instrumented runs per arithmetic policy per fixture, alternating which policy runs first. Restore is outside timing. |
| Result | Four of four fixtures passed exact F32 seed and continuation comparisons with pinned Ollama, including the instrumented prefill and source decode runs. The legacy arithmetic matched its own uninstrumented readout. GPU stage attribution completed. |
| Disposition | Reusable diagnostic. Next candidates selected for screening. |

Both arms used the normal text append encoder with widths `2048 + 2044 + 3 + 1`. The first two commands ran packed bulks, the third ran the three width-stable tail rows, and the fourth performed the final prompt seed and vocabulary readout. The target replay omits worker orchestration, tokenization, sampling, streaming and prefix publication. The last qualified worker benchmark remains the source for end-to-end throughput. The legacy decode arm differs only in its choice of singleton arithmetic on the common stock-prefilled state. It is a diagnostic comparator for the regression.

The instrumented arm inserts compute encoder boundaries within the same command buffer to collect calibrated Metal timestamps. It preserves dispatch order and one completion per command. The three-row tail keeps one command with aggregate GPU time, since its fine-grained stages would exceed the sample buffer. The GPU spans exclude counter resolution and JSON construction. Host command time covers encoding, submission, execution and completion. Logit readback and the finiteness scan are timed separately in decode.

```sh
cargo +1.98.1 run --release --locked -p stark-engine --no-default-features \
    --features inkling-slice --example inkling_slice -- \
    --profile-stock-4k "$ORIGINAL_SHARD" "$DRAFT_SHARD" "$STOCK_ORACLE"
```

Pass additional private stock-oracle directories to alternate the prefill run order. The oracle must contain the seed at 4,095 and its following teacher rows. The diagnostic emits only aggregate counts and timings. [Capture protocol](../archive/experiments/2026-11-21-inkling-oracle/README.md#stock-schedule-4k-text-capture) · [Worker baseline](../benchmarks/2026-11-21-inkling-mini-native-worker.md#source-aligned-m5-max-prefill)

## Stock prefill budget

Uninstrumented replay wall time including seed readback took a median **5.829 s**, range 4.883–5.924 s, equivalent to **702.72 tok/s** at the median of per-replay rates. Each prefill executed 12,870 dispatches across four commands (2,675, 2,675, 5,637 and 1,883). Median instrumented command GPU time was **5.835 s**. The GPU-time ratio of instrumented to uninstrumented replay ranged from 0.990 to 1.200 across fixtures, with a median of 1.004. Treat the stage split as attribution and the uninstrumented replay as the timing baseline.

| GPU phase | Median time | Share of instrumented command GPU time |
| --- | ---: | ---: |
| NVFP4 gate/up matrix product | 1.529 s | 26.2% |
| NVFP4 down matrix product | 1.250 s | 21.4% |
| MXFP8 routed experts | 1.490 s | 25.7% |
| Dense attention projections | 0.638 s | 11.0% |
| Relative/cache attention | 0.252 s | 4.3% |
| NVFP4 routing and bucketing | 0.186 s | 3.2% |
| NVFP4 quantization and ingress | 0.076 s | 1.3% |
| NVFP4 standalone epilogues | 0.094 s | 1.6% |
| Three-row tail, timed as one command | 0.097 s | 1.7% |

Medians of phase and share use the four per-input observations. The table selects phases, so it omits residuals, controllers, readout and command gaps. The relative/cache band includes 0.114 s local attention, 0.058 s global attention product and 0.017 s global mask generation. Global mask generation is roughly 0.3% of command GPU time.

**First prefill candidate.** Work on the shared grouped NVFP4 matrix kernel, which accounts for **2.78 s, or 47.7%**, across gate/up and down. First count live tiles, rows per expert run and exactness replay entries on the stock routes. Use a diagnostic guard/replay control on frozen operands to bound its cost before touching the production path. The first exact candidate to screen is to compute each weight column's scale-derived peak and binary quantum during the existing tile decode and share them across the two SIMD row halves. It can remove repeated packed-scale reads and calculations while preserving the guard bound, tensor accumulation and FP32 replay. If scale/guard work proves immaterial, use tile occupancy and traffic counters to choose the next matrix candidate. A 20% saving across the two matrix products is about 0.56 s of GPU time before whole-worker effects. Reaching 800 tok/s from the previous 6 s worker median would require about 0.88 s across the complete request prefill boundary. [Current matrix kernel](../../crates/stark-engine/src/engine/inkling_slice/grouped_pilot.metal) · [Rejected earlier kernel screens](2026-11-21-inkling-mini-packed-prefill.md#kernel-screens)

## Singleton decode budget

The uninstrumented source command took **33.40 ms** at the median (31.43–34.83 ms), versus **30.82 ms** under legacy arithmetic on the same stock state. The median of the eight paired wall differences was **+2.39 ms**. The GPU paired difference was **+2.42 ms**. Both paths used 1,883 dispatches, one command and one wait. Source host command encoding took 0.53 ms, and the full 200,058-value readback with the host finiteness scan took 0.072 ms. Decode stage timestamps increased source command wall time to 37.70 ms, so use them for attribution.

| GPU phase | Source median | Legacy median | Median of paired source minus legacy |
| --- | ---: | ---: | ---: |
| Router projection on 44 NVFP4 visits | 3.235 ms | 0.957 ms | **+2.279 ms** |
| Route selection and weights on those visits | 2.253 ms | 2.296 ms | −0.057 ms |
| NVFP4 matrices, including the fused singleton gate/up activation | 11.004 ms | 11.058 ms | −0.019 ms |
| Dense attention projections | 5.843 ms | 5.870 ms | +0.014 ms |
| Relative table | 0.225 ms | 0.140 ms | +0.085 ms |
| Vocabulary readout and final preparation | 2.147 ms | 2.131 ms | +0.009 ms |
| Whole profiled command GPU time | 35.343 ms | 32.988 ms | **+2.133 ms** |

Across separate aggregate measures, the router projection accounts for the observed GPU regression. The fused selection and weight kernel survived the earlier correctness and dispatch reduction work and did not account for the regression here.

**First decode candidate.** Specialize the source-shaped router projection for one live row. The current router launches four SIMD groups, each executing a 16-row by 32-column tile with one live row. Screen a narrower projection tile with more independent column groups, and stage or prepack router operands to reduce per-step loads and conversions. Keep the 16-wide accumulation order inside each of the existing 2,048- and 256-wide products and the final FP32 addition order. Compare against the existing source kernel on captured router inputs before changing the worker. Recover about 1.8 ms per worker token as the practical target. Require exact router logits, route IDs and weights, short and stock 4K logits, committed state and prefix restore. Rerun paired complete worker requests before adopting a replacement. The earlier serial DSpark verifier remains a separate workstream. [Source router kernel](../../crates/stark-engine/src/engine/inkling_slice/prefill_dense_source.metal) · [Decode follow-up](../INKLING_2_MINI_PERFORMANCE_FOLLOWUP.md#decode)

## Current worker throughput check

A separate session replayed the same four prompts through the sealed source worker from the earlier paired benchmark. It used the same binary SHA-256 `9000a3c1568b8dc2004d3a6b0e01ed28d37d3192b0bca560284e050ec34e764f`, context 4,352, vision and DSpark resident, drafting off and zero input reuse. All four emitted 256 tokens and reproduced the previous output hashes.

| Mode | Requests | Cold prefill tok/s | Useful decode tok/s |
| --- | ---: | ---: | ---: |
| Greedy | 2 | 834.63 (788.18–881.08) | 32.87 (32.68–33.07) |
| Seeded sampled | 2 | 846.81 (842.13–851.49) | 31.32 (31.24–31.41) |

The four-request rates reflect this session on the same binary. The earlier eight-pair comparison remains the promotion evidence. The worker replay had zero pageout and swapout deltas and no reported thermal or performance warning in the before and after snapshots. Two earlier client launches failed readiness and produced no measurements. The successful launch reused the same worker and artifact's identity-bound image check from a private copy of the recorded check store. The ready contract advertised both vision and DSpark. Worker replay aggregate SHA-256 `40b3f7812ae45e46b49f0b14084e9ac62d014022f2c5e2894b0c71e2cfcbe5a9`. [Earlier paired worker benchmark](../benchmarks/2026-11-21-inkling-mini-native-worker.md#source-aligned-m5-max-prefill)

## Attempt history and evidence

- The initial diagnostic requested more than 4,600 counter samples per command. The timestamp buffer allocation failed before measurement. The bounded retry used 2,048 samples and timed the three-row tail as a whole. Coarse timing on all four fixtures passed the pinned checks and assigned 47.6% of prefill GPU time to the two NVFP4 matrix products. It showed a +2.17 ms paired singleton routing difference. The finer profile then split gate/up from down and the singleton router projection from selection and normalization.
- The finer profile aggregate SHA-256 is `44b60ff9a3cbdb257a49321972c4252507099b53295f98ad20e8b5fdebffc15e`, with host aggregate `dd6317ef77588643f97f86a6d340a04c71944feac5f181e418d371440e6b9ecf`. The coarse profile executable and aggregate are `626f3ffeb90f17ae820256bd27958114519a094c99df62ff4206b34a608f3c41` and `4410d955393428afcf9872b36d043c6967e52d7b968f00d75639be4526372873`. Private receipts stay in the ignored stock-profile run directory. The stock inputs reuse the four pinned captures already bound by the source-alignment study. [Pinned receipt hashes](2026-11-21-inkling-mini-packed-prefill.md#20261121-inkprefill-09--source-aligned-worker)
