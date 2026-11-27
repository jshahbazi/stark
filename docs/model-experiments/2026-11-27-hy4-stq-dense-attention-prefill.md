# HY4STQ-PERF-02 · Dense attention prefill on Hy4 STQ

The starting worker is `f95cb83080827cde638adf65c6fdb2b039aaeedc`. It loads the original
Sushi STQ 4bpw pack at `61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`. The prior
fresh 27K prefill took 36.37 s, about 5.1 s longer than the same-cap Sushi run.
The goal was to recover that time in dense attention and retain the five-draft
decode path, flexible context and image-loaded prefix reuse.
[Previous OpenCode comparison and pack identities](2026-11-27-hy4-stq-open-code-performance.md)

GPU work held the shared Hy4 lock on the M5 Max 128 GiB. OpenCode 2.4.2
uses the same 16,229-token, continued 27,179-token and fresh 27,229-token
fixtures. They use 12 tools, 96 greedy outputs and the vision tower.
Stark fits its context within a 49,152-position benchmark cap and uses
4,096-row prefill chunks and F16 KV. Rates come from uninstrumented runs.
Prefill counts computed input rows. Decode counts the 95 output transitions.

## Projection attribution

The diagnostic sampler splits the existing encoder duration at the dense
attention projection boundaries. These times include projection preparation
and finish where present. Profiling and throughput use separate runs.

| Projection | Fresh 16K | Continued 27K | Fresh 27K |
| --- | ---: | ---: | ---: |
| Q_A | 0.196 s | 0.179 s | 0.357 s |
| Q_B | 0.811 s | 0.736 s | 1.452 s |
| Absorbed K_B | 0.785 s | 0.714 s | 1.411 s |
| KV_A | 0.256 s | 0.235 s | 0.463 s |
| IndexPool projections | 0.119 s | 0.110 s | 0.214 s |
| Absorbed V_B | 1.004 s | 0.920 s | 1.812 s |
| Attention gate | 2.010 s | 1.843 s | 3.652 s |
| Attention output | 2.375 s | 2.210 s | 4.516 s |
| Total | 7.555 s | 6.947 s | 13.876 s |

The six gate, output and Q_B shapes cover 9.62 s at 27K. Recovering all
5.1 s from them alone would require removing about half that observed time.
The absorbed K_B and V_B products account for another 3.22 s.

## Component screens

| Attempt | Result |
| --- | --- |
| HY4STQ-PERF-02a · Direct packed K32 | The 128-thread M16 × N32 × K32 product keeps two 16-row accumulator stripes and stores F16 in vectors. The six real banks at 1,024 and 4,096 rows matched all 283,115,520 F16 outputs. Nine alternating pairs per case measured 27–33% lower GPU time. At 4,096 rows, sliding gate went from 8.91 to 6.51 ms and sliding output from 9.91 to 7.08 ms. The paired repeat supersedes the wide-O timing from the exploratory sweep, where the control drifted. |
| HY4STQ-PERF-02b · Fuse gate finish and output prepare | The two Hadamard transforms keep their arithmetic and precision boundaries. The fused kernel combines the gate finish, sigmoid product and output projection prepare before writing F16. All 125,829,120 prepared values matched on deterministic inputs at both widths and row counts using release scales. At 4,096 rows the sparse chain went from 1.87 to 0.88 ms and the sliding chain from 3.02 to 1.00 ms. The first dispatch attempt failed on a missing Metal function-constant specialization and produced no valid timings. |
| HY4STQ-PERF-02c · Unpack and run register NAX | The best exact decoded repeat reduced sliding gate from 9.81 to 7.86 ms and sliding output from 10.50 to 8.15 ms at 4,096 rows. It needed 96 MiB of scratch. The shared-memory K64 products regressed and the packed candidate won the clean timing comparison. The first decoded arm failed to compile because of a reserved Metal identifier. [Pinned upstream decoded path](https://github.com/beamivalice/sushi/blob/v1.8.1/docs/hy4-flash-arch.md) |
| HY4STQ-PERF-02d · Partial final row tiles | The fast K32 product handles the complete row tiles and the existing K16 product handles the final partial tile. All 647,792,640 values matched across six projections and 1,025, 2,653, 3,941 and 4,096 rows. Padding canaries remained intact. Odd-row GPU time fell 23–30% against stock, and the 4,096-row arms remained within 2% of the prior wide kernel. |
| HY4STQ-PERF-02e · Compensated absorbed K_B and V_B | A 16-row, 128-channel tile with separate F32 accumulators for two F16 activation terms measured 18–19% less GPU time for K_B and 25% less for V_B at 4,096 rows. Across 28 actual-weight cases, worst relative RMS was 2.04e-7 versus sampled F64 and 4.55e-7 versus the existing F32 product. The fixed input scale was tested on synthetic values through ±4. The one-term arm had about 1.85e-4 relative RMS and stayed out. Preconverted fragment-packed weights would duplicate about 1,008 MiB across the trunk and stayed out. |
| HY4STQ-PERF-02f · Reorder threadgroups across row tiles | Swizzles grouping two, four, eight and sixteen row tiles preserved F16 output and raised sliding output-projection time by 5–28% at 4,096 rows. Q_B and the gate remained within 1.2% there. A full grid transpose initially reduced sliding output by about 11%. A targeted repeat found 3–4% for sliding output and 3–7% for sparse output. The full-request saving was small, so kept the current grid. All 932,981,760 checked outputs matched across the two row counts. |
| HY4STQ-PERF-02g · Exact absorbed-head register loads | Ten F32 tile and load variants matched all 3,875,880,960 active values across 120 cases on release K_B and V_B banks. All K_B RoPE tails remained intact. At 4,096 rows, the best direct BF16 load into F32 registers saved 12–13% on K_B and 9–10% on V_B with no new scratch. The staging tile variants were slower. The best direct path projects about 0.35 s of savings from the sampled 27K budget. Kept the incumbent heads and left the prototype out of the worker. |

The packed and fused paths retain the original F16 product boundary and
F32 accumulation. Their switches apply to wide attention prefill. Narrow
SPEC1 split-K, PAIR1 target routing and DFly2 verification keep their
existing pipelines. The packed kernel uses the same model weights and
existing scratch. Neither path changes the cache layout or prefix key.

## OpenCode and state checks

Each rate pair is prefill and useful decode in tok/s. The first exact arm
used K32 for complete chunks and fell back on other chunk widths. It
reproduced the input and output hashes and all three draft counts from
the prior repeat.

| Fixture | Prior Stark repeat | First exact arm | Partial tiles plus compensated heads | Exact product with partial tiles |
| --- | ---: | ---: | ---: | ---: |
| Fresh 16K | 671.57 / 49.99 | 801.84 / 48.40 | 840.88 / 44.95 | 838.31 / 48.31 |
| Continued 27K | 747.43 / 54.40 | 789.67 / 51.27 | 814.99 / 43.79 | 794.55 / 51.16 |
| Fresh 27K | 748.60 / 49.93 | 773.45 / 46.29 | 796.50 / 43.72 | 796.63 / 46.47 |

The compensated trial changed the output and reduced draft acceptance in
the 16K fixture from 70 accepted in 25 rounds to 68 accepted in 27 rounds.
All three useful decode rates declined relative to the prior repeat. The
component tolerance and exact BF16 weight cast did not establish output
parity. The compensated shader and worker option were removed. The
OpenCode continuation includes each arm's first answer, so the compensated
arm also rendered different continuation input.

The first exact arm cleared the prefix on image input and suppressed image
drafting. Its 27K continuation reused 12,288 tokens and replayed from an
empty cache with identical output. Under one loaded 49,152-position worker,
the 4,426-token fixture produced the same eight output IDs at 8,192 and
12,288 positions. Reuse across requests at 8,192 was 4,096 rows. The next
request at 12,288 started at zero reuse. A control startup was interrupted
before requests so the GPU lock could pass to the pending component retry.
It yielded no timing evidence.

The exact partial-tile run reproduced the prior three outputs and the
continued request retained 12,288 tokens. The fresh 27K prefill fell from
36.37 s in the earlier record to 34.18 s here. This is a 2.19 s observed
difference across two sessions. The client took 22.11, 21.26 and 36.90 s
across the three turns.

A nearby repeat used the exact same worker binary with wide attention off.
The two processes matched all three input and output hashes and draft
counts, with 12,288 reused tokens in each continued request.

| Fixture | Wide attention off, tok/s | Wide attention on, tok/s | Prefill seconds off → on |
| --- | ---: | ---: | ---: |
| Fresh 16K | 773.26 / 48.95 | 838.31 / 48.31 | 20.99 → 19.36 |
| Continued 27K | 742.86 / 51.80 | 794.55 / 51.16 | 20.05 → 18.74 |
| Fresh 27K | 742.55 / 48.13 | 796.63 / 46.47 | 36.67 → 34.18 |

Fresh 27K gained 7.3% prefill throughput and saved 2.49 s in this
same-binary pair. Useful decode measured 1.3%, 1.2% and 3.5% lower across
these three requests. One process per arm establishes the observed timing
difference with identical graphs in decode and identical acceptance counts.

A separate Sushi repeat at the same cap observed 1,048.22 / 53.95,
954.45 / 47.05 and 977.89 / 50.13 tok/s. It used its own template and KV8
and reused zero rows in its continued request. Its fresh 27K prefill
took 27.58 s, leaving a 6.60 s gap on that repeat and a 2.91 s gap to the
earlier 31.27 s reference. The five-second recovery remains open. The
instrumented control also put 10.91 s in routed experts and 6.10 s in MLA,
which have more time available than another absorbed-head retile.

The exact partial-tile worker repeated the image and flexible-window
checks from the first exact arm, with the same output hashes and reuse
sequence. Memory pressure stayed normal and swap held at 432.75 MiB.
With the existing [native STQ validation settings](2026-11-27-hy4-stq-open-code-performance.md#uninstrumented-opencode-runs),
`STARK_HY4_STQ_WIDE_ATTENTION=1` selects the exact packed and fused
paths. It remains an opt-in on the validation branch.
The release worker built with Rust 1.98.1. Release Clippy passed, the
native Rust test run passed 183 tests with six ignored, and the focused
Python suite passed 726 tests. Rust format, Ruff and diff checks passed.

## Evidence identities

| Receipt | SHA-256 |
| --- | --- |
| Projection profile worker | `5db59d8a682546dd6f80c51bd357805752f0c642616199890e626a74b5841ffb` |
| Projection profile aggregate | `cadf4fc996fbddf6e4fa802196ab7a806bb53d7b39cb5f3115f1839b5a61d71e` |
| First exact worker | `798246091c1c525fb60b34c75e33530f5352175b1247ff833f10eef750c423ae` |
| First exact OpenCode and lifecycle receipt | `d22fb2b0cfd0cca3dc9ba7912938832226ca90fe01c9af0358ce157d8c7437e8` |
| Exact partial-tile worker | `118c448a83c7010e415e56ca0c6865082f8dda1df90993ac5c1258f9c56318be` |
| Exact partial-tile OpenCode and lifecycle receipt | `6f37f328c713211530feb762955eff2372c8dbd1f2ce0ca49c9d6ab5566cfab9` |
| Same-binary OpenCode control receipt | `a09a8f75af5f8245811c3e58a3dcf9851a711c7b57743e7bef8a51edb42055b2` |
| Combined compensated worker | `f57e343fb109151f5964d42907ea072f44733027489bd0f217d9feb70da3768e` |
| Combined compensated OpenCode receipt | `17a9550f5017e1ea01c54322696b08fbfd4cfa4906bc6da64bd41d5fdd9184d4` |
| Packed K32 screen | `a597edbacce8b2afa2f501298d2934e4cb8d18aa715b0c7125be5d2ed1f1526b` |
| Fused gate screen | `7dd5c41142a5e5b29e6888193e157c74991ad3e0816af2e4d1a86fbbe1d1220a` |
| Masked wide tail screen | `333b1075eda87e110c0d64feb1b58e51abe00d124b0ab7b0ea4a629bb92688bf` |
| Compensated head screen | `58a7dfc86a5e9acf0fe4c4ba1d9aa49759ea52bc515544db716859e80b76febe` |
| All 42 trunk head banks, scaled-F16 weight round trip | `69091b2b39b1ed7f03efce37dd1e0c91a33e4e6adca71d47311366041a928236` |
| Decoded register screen, repeat | `6a1e537dd68a50936d5be374f5c999a773bc9228d5dd8545296d9a8f2867f801` |
| Packed grid-order screen | `d2c7016acc6a35f3c79f12548f44abf6702ed8a82ae62f237403fb54ff0bb7a5` |
| Packed output grid transpose repeat | `697655a4c6bfc181c1418b49c95878fd1817cab006762e2633f2a21266664aca` |
| Exact absorbed-head screen | `c910339c5c40cf924001c37591b96e1cad15dc04e1556b6cd2f86a564d98a8a8` |
| Sushi 49,152-cap follow-up | `016c57e8360209d1c46e8da016015ac6c0d6913e51e00df7535c451e253fb5a1` |
