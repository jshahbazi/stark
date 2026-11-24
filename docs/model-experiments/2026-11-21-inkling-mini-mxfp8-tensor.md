# 20261121-INKPREFILL-11 · Inkling 2 Mini tensor MXFP8 prefill

## Identity and controls

The target remains **1,000 input tok/s** for a cold 4,096-token worker prefill, equivalent to **4.096 s**, on the 128 GiB M5 Max. Both arms use the original three-pass checkpoint at revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb` and the same 64 × 32 grouped NVFP4 tile. The target SHA-256 is `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` and the DSpark SHA-256 is `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`. The source for the expected logits and residuals is the released, pinned Ollama capture with effective rows `2048 + 2044 + 3 + seed` and context 4,352. [Wider NVFP4 baseline and artifact controls](2026-11-21-inkling-mini-nvfp4-run-tiles.md) · [Source arithmetic and capture schedule](2026-11-21-inkling-mini-packed-prefill.md#20261121-inkprefill-09--source-aligned-worker)

The experiment starts at Stark `21987e88978e0fd47f49aae92469a7725f1ea89e` with its locked dependency graph, Rust 1.98.1, macOS 26.7.2 build 25G332 and Apple Metal 32023.883. Native profiles alternate candidate order within timestamped and ordinary commands and reuse one loaded model. Worker arms use one sealed build, resident vision and DSpark, drafting off, 256 emitted and sampled tokens, zero reuse, greedy and seeded temperature-1 sampling, and alternating paired request order with a two-second gap before each scored request.

## K64 kernel and numerical admission

The four MXFP8 visits use eight grouped products per bulk command. The candidate precomputes the L2 norm of each decoded expert weight row during workspace setup, then gathers the quantized activation rows into BF16 run order. A Metal tensor kernel forms the dot from independent BF16-input, float-output K64 partials. It keeps the rounded BF16 word when outward-rounded interval endpoints agree on its BF16 value and the interval stays away from zero. Inputs outside the finite, exactly materialized BF16 domain go to replay. Nonzero weight codes that underflow during materialization also go to replay.

With unit roundoff `u = 2^-24`, the admitted magnitudes and quantization grid keep scaled products and nonzero partial sums in the normal FP32 range. The source recurrence has at most 46 rounding additions on the gate/up path and 42 on down, using a conservative 31-add limit for a 32-lane sum. An independent K64 partial and the subsequent merges have at most 98 and 80 rounding additions. The sum of the corresponding forward-error factors is below `145u` and `123u`, respectively. Multiplying the norm product by `160u` and `136u` leaves margin for norm evaluation and outward rounding. The tensor argument depends on the M5's internal accumulation behaving as binary32 for these operands. Apple documents float destinations. The source audit checks the internal-precision assumption on the pinned workload. [Apple Metal Performance Primitives programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)

Each run stores one 32-bit replay mask per output column. Replay shares a weight column and scale stream across up to four selected routes while retaining the pinned FP32 source recurrence and SIMD reduction for each route. The original BF16 gate/up rounding, gated activation, routing weight, down-input quantization and route-order scatter follow on the same Metal command. The screening selector was `STARK_INKLING_EXPERIMENT_MXFP8_TENSOR=1` in the existing M5 Max source scope for grouped commands above 64 rows.

## Attempt history

| Trial | M5 Max result | Disposition |
| --- | --- | --- |
| Independent K64 dot with scalar replay queue and a `256u` norm bound | The two bulk commands had 41.27–49.97% ambiguous results per dot. Each exceeded the bounded 8,388,608-entry queue and selected the full grouped source projection. Pinned readouts passed. MXFP8 tensor and ingress took 128 ms and full source fallback took 1.310 s. | Rejected. A global list sized for sparse errors could not amortize this bound. |
| Column masks with a transposed cooperative-tensor store | Stock seed diverged. The first visible full-model residual divergence was at visit index 45. | Rejected. |
| Row-major cooperative-tensor store with per-column masks | Both exact 4K profile inputs passed. MXFP8 fell from 1.375 to 0.938 s and 1.390 to 0.963 s across the attached stage profiles. | Advanced to direct audit and worker timing. |

## Exactness and footprint

The native candidate reproduced the connected short and 4K aggregates. The short full-model run matched the pinned F32 logits, native cache, draft taps and snapshot replay. Four stock 4K runs reproduced **16 F32 logit vectors**, **768 BF16 visit residual blocks** and four snapshot replays. In a separate four-input shadow audit the source recurrence recomputed the tensor results on the same quantized ingress. The audit compared **3,295,567,508 certified BF16 words** across 4,827,119,616 total outputs and found **zero unequal words**. The remaining outputs used the source calculation. The audit performs extra exact work and has its own executable mode.

The full-target allocator delta rose from 1,245.03 MiB to 1,259.72 MiB, an observed **14.69 MiB** addition inside the 1,344 MiB source-prefill reservation. The weight norms are immutable per load. The mask table and the products are command scratch. Snapshot replay covers the completed cache boundary. The stock sequential decode and vision/DSpark paths use their existing kernels.

## Worker results

All eight 4K worker pairs passed the work and exact-output controls. Rates are arm medians. Gains are medians of paired candidate/control rate ratios.

| Sampling | Grouped source | K64 tensor and masked replay | Paired gain | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 824.10 tok/s | **914.14 tok/s** | **11.04%** | 4/4 |
| Seeded sampled | 808.96 tok/s | **894.10 tok/s** | **10.80%** | 4/4 |

Candidate prefill medians were 4.481 s and 4.582 s. The best request reached 927.21 tok/s. The gap to 4.096 s is **385–486 ms** at the two medians. Complete request wall time improved by 4.35% greedy and 2.57% sampled. Useful decode moved +0.07% and −0.54% in paired medians. In the two attached target-only comparisons, whole prefill improved from 4.329 to 4.147 s and 5.225 to 4.757 s. The faster target-only candidate reached about 988 tok/s.

The workers accounted for 98.00% of the sampled Metal counter increments across 116 intervals during the worker windows. All 135 host observations were fair. System pageouts rose by 555 pages of 16 KiB and swapouts stayed flat. macOS reported no thermal or performance warnings. The paired worker change and the two arm medians describe this session.


## 20261121-INKPREFILL-12 · K32 compensation and vector replay

The K32 candidate adds each independent partial with an error-free FP32 TwoSum merge and accumulates the residual separately. On the admitted binary grid, the final correction and the rounded residual sum allow two rounding units for the merge error. With up to 31 additions inside a K32 partial and the existing source depths of 46 and 42, the candidate uses conservative coefficients of `88u` and `84u`. The replay kernel reads four adjacent input values and four E4M3 codes per load. It multiplies and adds the products in source order.

| Candidate | Uninstrumented stock → candidate wall, two 4K inputs | Instrumented MXFP8, control → candidate | Stock readout |
| --- | --- | --- | --- |
| K32 and compensated sum, scalar weight and input loads | 5.472 → 4.662 s, 6.093 → 5.178 s | 1.472 → 0.807 s, 1.483 → 0.901 s | Exact |
| K64 with vector loads | 5.993 → 5.472 s, 5.661 → 4.905 s | 1.569 → 0.986 s, 1.558 → 0.954 s | Exact |
| K32 with compensation and vector loads | 4.997 → 4.602 s, 5.974 → 4.978 s | 1.586 → 0.777 s, 1.652 → 0.814 s | Exact |

The K32 alternatives sent 19.13–24.99% of each product to source replay across the two inputs. The K64 vector load result varied across the sessions. The combined K32 and vector candidate had the lowest MXFP8 stage times. A separate four-input audit compared **3,781,642,862 certified BF16 words** and found **zero differences** against the pinned recurrence. The rebuilt candidate passed the connected short and 4K gate and all four stock full-model controls, reproducing 16 F32 logit vectors, 768 BF16 visit residual blocks and four snapshot replays. The sealed worker restored a 2,048-token prefix, matched a cold changed-tail response, rejected saved state at both invalid packed boundaries and cleared it on cancellation. All 24 generated tokens matched at packed-tail widths 16, 47 and 64. Worker readiness verified the image path and DSpark admission.

The eight-pair K32 screen used the sealed worker and the same original source-mode controls as K64. All eight candidate requests cleared 1,000 tok/s and all paired output hashes matched. The greedy arm medians were 918.88 → **1,042.59 tok/s**, with a 13.41% median paired gain. Seeded sampled medians were 920.01 → **1,043.99 tok/s**, with a 13.06% median paired gain. Candidate medians were 3.929 s and 3.924 s. Useful decode moved −0.34% and −0.24% in the matched pairs. The workers accounted for 99.88% of sampled GPU counter increments across 109 intervals, with nominal and fair thermal observations. Pageouts rose by 53 pages and swapouts stayed flat. Model-free repository validation overlapped part of this short screen. The longer session provides the sustained result.

## Sustained K32 worker benchmark

A separate session completed eight eligible pairs per sampling mode, with no builds or other benchmark work during the request windows. Both arms used the 64 × 32 NVFP4 option. The candidate also used `STARK_INKLING_EXPERIMENT_MXFP8_TENSOR=1`. All 32 requests had the prescribed token counts and zero input reuse. Output hashes agreed in all 16 pairs.

| Sampling | Grouped source prefill | K32 tensor prefill | Paired prefill gain | Candidate range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 907.18 tok/s | **1,058.20 tok/s** | +17.01% | 1,025.66–1,064.20 | 8/8 |
| Seeded sampled | 902.73 tok/s | **1,053.94 tok/s** | +16.16% | 1,047.49–1,065.23 | 8/8 |

The candidate met the 1,000 tok/s median goal in both modes. Median candidate prefill was 3.871 s greedy and 3.886 s seeded sampled. The 1,000 tok/s threshold was met on 8/8 greedy and 8/8 seeded sampled requests. Median paired end-to-end wall changes were -5.45% and -5.05%. Useful decode medians were 33.10 → 33.13 tok/s and 31.41 → 31.38 tok/s, with median paired rate changes of +0.08% and -0.05%.

The workers accounted for 99.87% of sampled GPU increments across 219 intervals. The host samples included nominal and fair thermal observations. Pageouts and swapouts stayed flat. The compensated K32 path is available as an opt-in M5 Max source-prefill experiment on this branch. The measured configuration sets both `STARK_INKLING_EXPERIMENT_NVFP4_ROWS64=1` and `STARK_INKLING_EXPERIMENT_MXFP8_TENSOR=1`.

## 20261121-INKPREFILL-13 · NVFP4 lane scale sharing on 64 × 32

Adjacent lanes that decode the two words of one 16-value NVFP4 block shared its scale with a SIMD shuffle. The workgroup published the two half-block peaks and quanta for all four row fragments. Both stock seed and first continuation checks passed. Compared with the same loaded 64 × 32 incumbent, whole ordinary target time was 4.344 → 4.665 s and 5.195 → 5.143 s. The NVFP4 product times in the instrumented arms were 2.190 → 2.293 s and 2.191 → 2.189 s. The tile combination was rejected after its mixed whole-target results. The prior 32 × 64 shared-scale screen remains separate. [Original scale-sharing experiment](2026-11-21-inkling-mini-nvfp4-router.md)
## Evidence identities

| Artifact or private receipt | SHA-256 |
| --- | --- |
| Locked dependencies | `07b6d42497825a651719685a28a107bfc0a32119b8e8b8adeab2d6c01a96695b` |
| Sealed K64 native example | `cc2cd0c646fcce33d0a79a952ef83a819cd5a0b5186794cdc1ae8db144736370` |
| Sealed K64 generation worker | `49c5e021d38fe7afb6a7fb6970696a34b2dc8a63e4ae39f0b63e0440922b7b7c` |
| Scalar replay and overflow profile | `4a4782a54d304275362310b26186355c691713db471ef086dc7599bfd73b420f` |
| Rejected transposed-store target replay | `1f64788fe7c61c63b84b0306a03ccd6905391b20093cadffdb109a483b950f0c` |
| K64 column-mask stage profile | `a26233d13429f3dc912f9b07c8cbfb184da9fabd18d805d89688ff6283b43581` |
| K64 four-input certificate audit | `71eb73049566f827a7f44d7982b3146cb9f34ed806e234edbaaeb62de9d39f95` |
| K64 worker pairs | `3b56b31fc18c787d7628245ad2ecc70347e3c6f9e5f283a315f9855ec176b525` |
| K64 worker host observations | `b76075046ac4e5cb35e3282e5cf1b1d53f64c0817b30320406d4e6030c69ee4d` |
| Paired worker harness | `6d80cd242c5466d779dedeff31788b0489ced537bac9f5a96dc2a7fe9e36d2a1` |
| Compensated K32 stage profile | `014f3618eab98d8ce44a59e9267e55a8b009364c55f7470393a170d5fda553f0` |
| K64 vector-load stage profile | `d673a9c53c21e26c20d9f881aaf82eb30efc4e73d512b946678d60bd11392507` |
| Combined K32 and vector-load stage profile | `bda3783cc61bc1ea30622f0a8bf781f21d20bb4a2c9e822072aac58158d898eb` |
| K32 four-input certificate audit | `f482ce93fbdf2ca51818e06096b9d5344e151b13d1dad8e3d46e3ac1bfbd37f8` |
| NVFP4 wide lane-scale profile | `842838815cb23456608ac44eed4c667a5c0e684f06a4616f4aa79f31f77e549a` |
| Sealed K32 native example | `4176d7fe34e709a18b8c9540b0bb2a8ae0abde359447948d03a42a8621979784` |
| Sealed K32 generation worker | `bc5de5f5336093eb27c82ade44b9422f720f4a4035c08a8e498ba05d1f407fb9` |
| K32 worker screening pairs | `eb49c582668332332f0422ff8cb4be5eab0562a8ea2e00d714da868bb665d7d1` |
| K32 worker screening host observations | `be13a77225a338b0c775f32658fad38890feff3e776ea8608f06ade4224fe186` |
| K32 worker prefix and cancellation controls | `d430d13ea5466c0887ec46e83458ff0208493290492839462df03023da6e0477` |
| K32 worker short packed-tail controls | `f431d8684afff1117d665c127767293d734d8ef35db4b5c4763f908ae5b81da5` |
| Sustained K32 worker pairs | `339fc29df1f8e9cc9facf8a1422698c7d5dcc2046f8c911b245d1253945c52b6` |
| Sustained K32 worker host observations | `13eb6cb82c066c58aeb2655a3732daddb5bf1fbb05c5a9fbd2f468f20db4739d` |
