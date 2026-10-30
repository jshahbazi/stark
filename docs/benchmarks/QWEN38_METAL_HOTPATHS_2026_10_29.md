# Qwen3.8 Q4-XL Metal performance note

Aggregate measurements from 2026-10-29 on the M5 Max 40-GPU-core, 128 GB host. Candidate workspaces, raw receipts and model data remain private and ignored. This study extends the earlier source-bound records.

## Scope and workload

The measurements use the same-quantization Qwen3.8-Flash-Next Lily Q4-XL workload: 4,096 input tokens, 256 requested output tokens, context 4,352, temperature 1, top-p 0.95, seed 42. Top-k 20 and nucleus-only sampling are reported separately. Speculation, MTP, and vision are disabled. The precision policy is BF16 activations, F32 reductions/logits/GDN state, and Q8 attention K/V.

The initial quiet-window worker is bound to source revision `f669f721d57612fb1316a26e462432476c95b458`. Experimental candidate snapshots retain that base unless their manifest says otherwise. The HC literal-down control rechecked production build inputs against current-main revision `f7ff311be3187de705531089b2e3081bbf001891` while retaining its candidate snapshot from `f669f721`; the branch at aggregation was `codex/qwen38-metal-hotpaths` at `f7ff311`. Each private variant has its own source manifest and patch hash in the evidence list below. Do not infer a shared binary base across experiments.

## Quiet baseline

The baseline completed 20 requests. We exclude the first warmup for each case and retain nine measured requests per case. Decode rate is `255 / decode_seconds`; pooled rate is total 255-step decode work divided by summed decode time. No slow requests are filtered.

| Sampling case | Measured requests | Pooled tok/s | Median tok/s | Range tok/s |
|---|---:|---:|---:|---:|
| Top-k 20 | 9 | 73.62 | 73.60 | 71.29–75.81 |
| Nucleus | 9 | 69.85 | 68.94 | 68.23–72.58 |

Every request returned 256 tokens and matched the expected per-case output hash. The measured requests had no pageouts or swapouts. Clock samples fully contained by each request averaged 1,305 MHz for top-k 20 and 1,324 MHz for nucleus, with active ratio near 1.0 and 81–83% clock-interval coverage. This documents a comparatively steady baseline window; it does not prove that all unrelated host activity was absent.

The 77–80 tok/s desktop target was not met by this baseline. Later candidate measurements do not establish a reliable improvement to that target.

This quiet-window receipt extends the [fixed-quantization kernel study](QWEN38_KERNELS_2026_10_29.md) and [sampled BF16 profile](QWEN38_SAMPLED_BF16_2026_10_28.md). The earlier study's same-loaded control measured 61.1398→62.3204 tok/s top-k 20 (+1.93%) and 64.0755→64.1106 nucleus (+0.05%) for its retained graph; its individual windows also showed substantial drift. Those figures and their source bindings remain in that report. The separate windows cannot be pooled.

## Instrumented stage profiles

Stage profiles are diagnostic measurements, not throughput benchmarks. Each profile run has three exact 256-token replays; replay zero is excluded. The top-k profile averages two measured replays at 13.659 ms per decode step of measured GPU span. A later canonical nucleus profile also passed three exact replays and averages the two measured rows at 14.607 ms per step. Instrumentation and different scheduling windows make profile spans unsuitable as production throughput claims.

| GPU phase | Top-k 20 ms/step | Top-k share | Nucleus ms/step | Nucleus share |
|---|---:|---:|---:|---:|
| MoE | 4.072 | 29.81% | 4.423 | 30.28% |
| Hyperconnections | 3.230 | 23.65% | 3.338 | 22.85% |
| GDN | 2.963 | 21.69% | 3.192 | 21.85% |
| QSA | 2.147 | 15.72% | 2.298 | 15.73% |
| Vocabulary head | 1.130 | 8.27% | 1.223 | 8.37% |
| PLE, embedding, draw | 0.117 | 0.86% | 0.134 | 0.92% |
| **Measured GPU span** | **13.659** | **100%** | **14.607** | **100%** |

For the top-k profile, CPU command encoding averaged 0.486 ms per decode step, blocking wait 13.396 ms per step, and sampling 0.136 ms per sample call (selection 0.078 ms, finite validation 0.058 ms). In the nucleus profile, two measured replays contribute 510 decode steps and 512 sample calls: encoding averaged 0.915 ms per step; blocking wait averaged 14.149 ms per step; sampling averaged 1.529 ms per call (selection 0.909 ms, probability filtering 0.535 ms, finite validation 0.082 ms). Nucleus logit readback averaged 0.021 ms per sample call, logit validation 0.082 ms per call, and n-gram staging 0.071 ms per decode step.

The wait counter includes blocking GPU wait and has different measurement boundaries from GPU spans. Do not add it to GPU phase time or interpret the difference as an idle-gap estimate. The finite-guard candidate is evaluated separately below with same-loaded controls. Its component timer uses elapsed wall time, which includes OS descheduling and is not a thread-CPU-cycle measurement.

## End-to-end paired controls

The prior HC gain-cache candidate replaced the production two-dispatch HC path with a three-dispatch gain/RMS prepass. In the same-loaded ABBA, its pooled rate fell from 74.21 to 72.68 tok/s for top-k 20 and 69.73 to 68.11 tok/s for nucleus. Its component timing also regressed from 33.627 to 40.932 µs/read. Reject this candidate.

The dynamic-pair MoE candidate completed a same-loaded 20-request ABBA with exact output hashes and 256-token lengths, but did not show reliable benefit. Each case/route has four measured requests after four warmups.

| Case | Reference pooled / median tok/s | Candidate pooled / median tok/s | Candidate range tok/s |
|---|---:|---:|---:|
| Top-k 20 | 65.13 / 62.81 | 63.07 / 62.86 | 59.50–67.65 |
| Nucleus | 61.69 / 60.01 | 61.93 / 62.17 | 60.20–63.25 |

Candidate-minus-reference ABBA cycle means were -4.99 and +0.40 tok/s for top-k 20, and -1.23 and +1.40 for nucleus. These CPU-encoded dispatch counters can include unsubmitted lookahead and do not prove GPU execution. The matching dynamic reduction depends on the current compiler's observed `simd_sum` order; Metal supplies no portable matching-order guarantee. Reject the dynamic-pair candidate.

Literal HC-down dimensions passed exact component controls. In a later same-loaded ABBA, pooled rates were 65.74→67.16 tok/s for top-k 20 (+2.2%) and 59.69→59.88 for nucleus (+0.3%). Cycle-mean deltas were +2.72/+0.31 and -0.15/+0.56 tok/s, respectively. This is a small, mixed result, not evidence of a reliable end-to-end gain. The run had high external activity: request 20 recorded 180.24% weighted-mean external CPU, five external GPU clients with positive counters, and 100% device GPU utilization. Clock, temperature, and power correlations are descriptive; GPU utilization is not exclusive, GPU counter units are unverified and may overlap, and no client identifiers were persisted. Do not attribute the run's falling reference rates or route differences to a named process or a single cause.

The finite-guard candidate preserves the existing predicate and enables NEON vectorization on this host. Control builds use Rust 1.98.1 with locked dependencies and separate release target directories. Two 20-request same-loaded controls passed exact output-hash and length checks: top-k 20 produced hash `621ca90417af420aaf9defdc957f7cc1f57f318c08d908be764834abb382b0d0`, and nucleus top-p 0.95 produced `9c33d3a4ef1fef97562e00f7f88b5dc287aacadb16b50c4739aec28dbea1b614`. Each case excluded four warmups and compared eight measured requests per arm. The guard's measured elapsed time per request fell from 38.73 to 8.36 ms mean for top-k 20 (35.52 to 7.97 ms median), and from 53.71 to 9.70 ms mean for nucleus (36.79 to 7.85 ms median). The nucleus serial mean is affected by two large per-request elapsed-time observations; elapsed time includes descheduling. These component results, plus exact predicate preservation, support retaining the finite-check implementation.

The paired end-to-end rates do not show a reliable gain. Top-k pooled/median rates were 67.86/68.49 tok/s serial and 67.30/67.72 chunked (-0.84% pooled, -1.12% median). Nucleus rates were 57.43/60.58 tok/s serial and 58.25/59.38 chunked (+1.42% pooled, -1.99% median); ABBA block directions were mixed in both cases. Treat these as noisy same-loaded observations, not a performance claim. The 77–80 tok/s target remains unmet.

## Component experiments

Component results do not predict full-model speed unless confirmed by a same-loaded end-to-end control.

| Candidate | Evidence | Result and disposition |
|---|---|---|
| HC literal dimensions v2 | 288 exact cases, 1,347,552 checked bytes, matching aggregate hashes. Streaming test: 97 distinct reads, 679,608,320 weight bytes, 16 samples. Same-loaded ABBA bridge completed; see paired result above. | Mean GPU time/read: down-only 26.339→22.447 µs (14.77% lower); up-only 23.927→24.031 µs (0.43% higher); down+up 25.039→22.185 µs (11.40% lower). The ABBA result was small and mixed; do not retain on component timing alone. |
| GDN register reuse v1 | Exact recurrence control failed at state `hk=2, h=6, gate=1, style=0, step=0`. | Reject v1. |
| GDN register reuse v2 | Corrected candidate passed 12 cases × 16 steps over 341,065,728 checked bytes; reference, production, and candidate SHA-256 all `9cafa233669f6eefa90f06022cca7a068d4d147ab637190ecc901961697a74b9`. | In a 36-layer, 16-sample component chain, median GPU time changed 16.862271→16.665507 µs/layer. The 0.197 µs/layer difference is about 0.0071 ms across 36 layers per decode step. This does not justify full-model retention on its own. |
| Packed-Q4 N64 gate/up | Nibble layout passed; all seven synthetic numerical styles were finite. Ordinary raw gate/up relative L2 was 4.99e-8/4.75e-8; ordinary activation, branch, and HC BF16 outputs matched bitwise. | 48-layer timing rose 65.125→238.329 µs/layer (3.659× slower). Reject unsplit N64. Synthetic controls are not model-quality evidence. |
| Packed-Q4 split-K8 gate/up | Nibble-layout and synthetic numerical controls passed, including finite partials. | 48-layer full component chain (split, reduction, SwiGLU, down/HC) rose 67.472→145.360 µs/layer (2.154× slower). Reject. No full-model integration. |
| Packed-Q4 N16 and N32 | Source manifests and patches exist. | Not built or tested. No result or recommendation about performance. |
| Grouped NAX MoE | Five synthetic numerical styles remained finite but showed relative L2 errors of 0.004925 activation, 0.007467 branch, and 0.007568 HC; BF16 agreement was roughly 55–70%. | Component medians were about 50% slower than generic in both intrinsic and routing-inclusive timing. Rejected for performance; model-quality acceptance was not attempted. |
| GS64 specialization | Two measured requests per case in separate windows after one warmup; no same-loaded route ABBA. | Top-k median 74.598 tok/s; nucleus 66.075 tok/s. Not a causal comparison and not qualified for retention. |
| Four-bank CPU radix histogram | Standalone exact control passed 10 finite-order cases. Sixteen AB/BA pairs on each synthetic 248,320-element input; output fingerprints matched. | Original→banked mean sort time was 544.888→617.852 µs for finite-random input (+13.4% slower) and 907.078→1,101.575 µs for a hot-low-digit input (+21.4% slower). Reject the four-bank candidate. This synthetic CPU component test is not model-level performance evidence. |
| Finite-value guard SIMD | Exact predicate retained; NEON SIMD finite check passed the same-loaded control in both selector cases. 20 requests per case, four warmups and 16 measured requests (eight per arm). | Guard elapsed time/request fell from 38.73→8.36 ms mean for top-k 20 and 53.71→9.70 ms for nucleus. Component improvement supports retaining this finite-check path. The paired end-to-end results are mixed and do not establish a reliable speedup. Timers use elapsed wall time, including OS descheduling. |

The initial virtual-lane reduction failed 5,570/16,384 rows, and a hardcoded shuffle order still failed 1,273 rows. The dynamic-offset version passed the device-bound controls before the paired full-model run above. Earlier masked-nibble and row-packing trials remain documented in the linked kernel studies.

## Status

The finite-value guard SIMD implementation is retained based on exact predicate preservation and faster guard elapsed time in both selector controls. Its end-to-end impact is not established: pooled rates moved -0.84% for top-k 20 and +1.42% for nucleus, with mixed medians and ABBA blocks. The literal HC-down end-to-end result is too small and environmentally noisy to support retention; the full-model bridge is complete and inconclusive. GDN component gain remains unproven end-to-end; packed-Q4 candidates measured slower; other MoE candidates were rejected for parity, quality, portability, or performance reasons. The 77–80 tok/s desktop target remains unmet.

## Retained implementation

Implementation commit: `28510ca72381834161fd1e159825daa39d5452e4`.

Only finite-value validation changes. The shared [predicate](../../crates/stark-engine/src/engine/hybrid/finite_guard.rs) checks 32-value chunks with a non-short-circuit boolean reduction, allowing LLVM to vectorize the comparison, then checks the scalar tail. It rejects the same infinities and NaNs as the reference and accepts signed zero, subnormals and finite extrema. Callers retain their existing empty-slice and shape checks. The two Lily readout guards and worker sampler use this helper; other generic validation paths remain unchanged.

The private same-loaded control adds equal TLS selection, call-count and elapsed-time instrumentation to both routes. The production helper contains no selector, counters, timer, new environment variable or protocol field. Its tests use the repository's custom test runner and cover short lengths, chunk boundaries, tails, IEEE edge classes and the full 248,320-value vocabulary. Weight quantization, inference precision, sampling arithmetic, RNG and Metal shaders are unchanged.

The 40-request control used `chunks_exact(32)`. The final helper uses the typed `as_chunks::<32>()` API required by Clippy. These source versions have separate validation records. A pinned Rust 1.98.1 differential probe passed 16,128 comparisons across empty/short inputs, chunk boundaries, seven nonfinite encodings, and 1×/4× vocabulary sizes. Both chunked versions have the same hot NEON scan instructions, with different loop setup and termination. Nine interleaved synthetic samples measured 63.125 µs scalar versus 14.500 µs for each chunked version at 248,320 values; at 993,280 values they measured 253.042, 57.500 and 57.333 µs. This qualifies the typed guard separately; the 40-request throughput observation remains bound to its original control source.

## Production validation

The final typed source passed 668 native CPU/Metal engine tests, seven CXX tests and three doctests using the pinned nightly test runner. The Rust 1.98.1 release workers passed 55 generator tests and 39 scoring/probe tests. There are 45 ignored engine tests and two ignored generator tests; these are excluded from the pass counts. Release worker Clippy with all targets and `-D warnings`, the repository's formatting check, repository policy checks and their 41 self-tests passed. The benchmark's local links and index entry were checked.

The production build uses its own Cargo target directory. Its frozen inventory covers 880 inputs and 18 internal CXX symlinks; only the four finite-validation source files differ from `f7ff311b`. The earlier inventory and binary audit remain preserved separately. The complete prior weight-hash receipt has 55 matching shards totaling 106,905,155,192 bytes; all 55 device/inode/size/mtime identities were rechecked unchanged, without rehashing model payloads.

The final release generator, SHA-256 `1dfd74cf6ac4c2d2bc143bb011b1e1c5240a152e240f823260b0383260df9db1`, passed three uninstrumented sampled replays per case. Each emitted 256 tokens and matched its expected output hash. Excluding the first warmup leaves only two measured requests per case:

| Case | Warm median tok/s | Measured request tok/s |
|---|---:|---:|
| Top-k 20 | 76.27 | 76.73, 75.82 |
| Nucleus | 71.34 | 72.12, 70.55 |

These smoke measurements verify the final implementation and remain a separate window from the quiet baseline and paired controls. They do not isolate a causal throughput gain or establish consistent 77–80 tok/s. The preceding `chunks_exact` production prototype also passed three replays per case, with warm medians 76.47/71.68 tok/s; its source inventory, binary audit and measurements remain separate.

The final source input-set SHA-256 is `c322966e46820faeabeab9f889b044d8305e0dd0e23301b917b63655190d338f`, with source-delta SHA-256 `2226d609f96fdf4082a16e252930d96658c0611c6a20a463d21d6777628e2df5`. The release audit rechecked every input and symlink, found all four root Metal sources embedded verbatim in both workers, and found none of the private trial hooks. The scorer SHA-256 is `f34dd0286db31320bcdf8c3c960e316324432fa7b5981010b6477096797cf2e6`.

## Evidence bindings

The table identifies aggregate receipts, clock captures, source manifests, and patches. Full SHA-256 values below bind the private evidence used for this draft. The source worktrees and raw JSONL/log data remain ignored and are not included here.

| Evidence | Receipt / log SHA-256 | Source / manifest / patch binding |
|---|---|---|
| Final typed production validation | Source receipt `d30c88c9814503e1718a4a648274bf0ba5488a5b6b5c6fccd3f46d94650e260d`; binary audit `c3d95d5b47c3fead3e742c67a6943ac3cb338c0d829bfa48ac0cd407fad766a8`; typed CPU probe log `30b0a553ac6104cf00de05dca2e9c7e361cf42965a0fe8d2dc76f2dd4ab12fb0` | Input-set and source-delta hashes above; 880 inputs, 18 symlinks, no Metal changes |
| Final typed production sampled smoke | Top-k receipt `c5a26fdd2df2d78d26eee0b3e4d844d7d738d3b1dd24e4247b357730bedf0f7e`; nucleus receipt `76b01b7f869e4be51a95e96d76e82d8429b3a25facef2230909a0d341dc8874d` | Uninstrumented generator `1dfd74cf6ac4c2d2bc143bb011b1e1c5240a152e240f823260b0383260df9db1`; six exact requests |
| Earlier chunks-exact production smoke | Top-k receipt `929ba46e36a821ac528d51ae04067e88f598eda31e5d7cbac37bcc2948db6904`; nucleus receipt `1bf30d3823045ba3c4b5a0c96e71c019b69cecd4efc0bbfde7542013408cb8cd` | Earlier source receipt `82f441da32f8b07d79f655189548e91ec1540f09136f759ef92c7b4d277f5bca`; earlier binary audit `4363852d071324147bf12752f1ecae797bb0ec11ec75140f705b9eaee903f480` |
| Quiet baseline, 20 requests | Receipt `4728f938f4d8b95b5c69cc239cc2127390ec874dbfa693fdbe91e4e44438307d`; clocks `5a6f0327285e957b2aed81f01c75f051e02a91dc87f1c25ec0dc36ec9ff25e66` | Worker `c909c5d7b5a9f460684aaf4dc9af8a114988874ed5c1db1da08f59b6d971426c`; source `f669f721d57612fb1316a26e462432476c95b458` |
| Top-k 20 stage profile | Run `87eb94019ea760b4c90bc6947eaab71b532827e57999333fb463548a3352caaa`; stages `24031b89fe0276e1b1747fdd84e5a7ea6cbb3c07e5a03a464e4bec7478e9afef` | Same sealed baseline worker above |
| Canonical nucleus stage profile | Run `7c26683291eb0a26fab7244a7e462c298504ef2e278d7078ee51554d7f41d65a`; stages `0c7c22fa20bc8c9af9aac29524c77a10beda34f593cbe10e23dacfedc2e2cf69` | Same sealed baseline worker above; all three replays exact; first excluded from profile aggregates |
| HC gain-cache ABBA | Receipt `edccec6a16c6ad6a99a121d1ec1b7ab3300bd313a5bf6df7ab1f1dcd3236a8d4` | Instrumentation manifest `4a7a802e6f76822b5f3cc0f886215d9ea84b03b2b01373906c0e261ff7acd9f6`; candidate manifest `a926fffad9766d134ba750ac772451f637f3c988dbcc3a6f3be5256dd54e8884`; patch `e2e8e5a5e51aa5d6f9471e7d26857f03d5a5403e1368e6f505e96780809721ed` |
| HC literal-down ABBA | Receipt `762c001be395dfa1e207f2512faed4c28788ec2f2d3075458eec5e273c998c73`; clocks `0af1caec1657930a62992ecbd25e00c4e5c84d612259a9d697bd08a32ac55922` | Control manifest `6c25a005ff0e09cf92d7be0476ff1b9d3439c967f2e282b8c11e3a1656a8907f`; source manifest `c2bab096cc7f4100ca9aa964693a4b582a5ed3aec9b8a1eddc0f99de305bbc46`; full patch `f9458b043432f9b61533f58e29bc49d0b1b5985cd523802044c4101d1eefba6e`; instrumentation patch `7ab0cdb0e524b4d8509e4ee0a0f0c46234f3f8691cb580bed590a5bc14f90bbf` |
| Dynamic-pair MoE ABBA | Receipt `51f24b52a4ef16ded450906c91c09ce054c497738f9b1627659b0adf0d61d469` | Control manifest `f25a85e20b554bb84bea3a48f93daf503e474fa6759b09a23d93a711d67d46ad`; source manifest `b74f07de19a65be3ca32c95fed8911d92819e4e245d290fd412a0d5d3acd0337`; candidate manifest `c4ae25c65258e49d6fcc779682c547152de3baa15260294d110944914e0951f4`; full patch `08d38ead4d7b0d1e9c70f2441f4c9bdb2bf613d73f5d0e21b418c7221c65eef4` |
| HC literal-dimension component | Exact `2c128b744b91ceaa5738e83d2c55425ffeb5bfdbca636197ee761db18b816977`; timing `2da4c62ef65414d180e04c25e51159794c20b1cba0b4637a44694f3379b98569` | Candidate manifest `ad1a82feee8923d872c72a71fc9c4482c4efc60a172fac34b825b3947d202955`; patch `5c255f7126fa2f7d384c45fa3ffc5cfbee42e21ab5fe73014ac4a8cfa867d8be` |
| GDN register reuse v1/v2 | v1 failure `d8883ddd5e9208bd86be1733e93f0b1ac834c1cd0fc0b6db1baa662594182202`; v2 exact `3ba5d825810b1a7fdc7218ef51413305894c20e2dfeb235787b0e0335886833e`; v2 timing `ae9499ba9620ad888d7ad4e3161ca41f5ac6fb556b573d89ffc166b25af6b911` | v2 manifest `268ca928a53c12446519bc317a4198196c8658baa2af8efe166559560b82acf1`; patch `9fb8acca46fd248b03d9b64d146af56f74f023374def83d09e1427138a34ed2a` |
| Packed-Q4 N64 | Nibble `de099182a03086ea591dd6a68a993e150f335dce0206e7fb068086ca7d0a646e`; numerics `6fc76217ddc0c096ee632b1ea98c587a1d05066a3045d474c86c380b047c03e7`; timing `274c894738843ae1b78f2784d1615f924cef9454a74c7c16e353cb121638539c` | Manifest `fedca61484059df069c6bce30f11c8c2c229e9ab0df454f2bd580f921d3e1b6d`; patch `1572248f1e3b8d9e6ccb274d6d02e25fb827e8d081cf7236d1b0c33dfdeaba05` |
| Packed-Q4 split-K8 | Nibble `3036bed98a6fcdcb19287934f6e1a473c750b482d64b83bb9796b4f894be10d9`; numerics `3428af4b7f00081b199ddd9330c2223f4f9103cf43155fc5103fa373b112ef10`; timing `820e148f97552f8f3acebe7efc59355f10dd7246a28d0d7a5dc23b4bfabb07e9` | Manifest `d0a93e44fade8e932683924005ed5ae3d013085d6530e95d02162dbbfec6ffa3`; patch `a36112fcd3eb38131c273aa84400e86769bff2ddb30c9dce613523ff5bec6369` |
| Packed-Q4 N16 / N32 (unrun) | No run receipts; source-only | N16 manifest/patch `91bca48837cf5badb8dfd849aef4dc6e5fcb624168d9de13871521d5a5ef9e00` / `1a614d2020e6f6d45699b19184ac0b5a239c8b833c33d79b573f392c5e689077`; N32 `bebc19ec1fa63c35b5151ed4a9b85ee0786635ef5ed1ac6d9b83ef2008922bd3` / `80c78baa6db5c5d7a43648ecfc159ac3c9b7765b52b6a401d837afea03e850ce` |
| Four-bank radix probe (rejected) | Exact/ABBA scalar log `f2a965b179ea6daab1d52f69a69bf57c8f3c3f5c3b6f1493ca0f71b43f6d81e6` | Standalone probe source SHA-256 `c582d74fdd6bbfa0ed27427c85aa597d8a0f3943d70675fc1ae40ff8b7b93b95`; no production source change |
| Finite-value guard SIMD, top-k 20 + nucleus controls | Top-k JSON/log/timers `0acc11490efcaf52ac31d8b7979792d7417f46282b0c64e4de89a86bde43c3a0` / `3b0d5f0244064d897a95bcc64b13700b99081c0d0bda882f4983860bb3447e47` / `a067873e149886e80a39ee39a7af741442fe7d60b8719b11eef6e60e120ce05d`; nucleus JSON/log/timers `bbef95c8407ec76ac01fc2c428c799f806418d40d8c52a8564d871014cf97fef` / `68da50e5fe599ff24ca4e4e3bd6f0b02ec9a940c48a1bad9863c902b3c93660d` / `257868ddbabe092d64c6187864a511a03960bb752cd2c02360d669dd981e02c1` | Shared worker/config/source receipt/corpus `654ce2b1c4399318409a8a4cec69bd82c30652a6a08b71b38de9a4e84c093b49` / `2a91155be52ebbb0f6dff20e61dcd8ae7484154a56b4afe067246c59da46f899` / `7f5c685f35fc4c5d1509f6379feb1a965cb625f7a1ce76a0b5fe69983561dd30` / `daecdf178c6429a721f9e1b5ae379c664adb8f8af55b381535355636ec469955`; control seals `6e8c38f80e37aa18de4ddfd31b4ca3f6477644376fd690df0670ec933c717e26`; control/production patches `44a8021c3a5c55bd75d7642632dee66b1aba94200266de95aa9a20618e47a5d6` / `0caf09af08eb27e32fbdfadad8b773f37fb393b417ad8b3e09141c7acfa95fa2`; harness source/assembly/checksum manifest `1e3df48f06922d004792ce09bcb18cc526ee25242917fc8bb47c141bab81a625` / `fd6cc970fc4c11a75e08273c84547b32c14d43b9dcb0d06e2d382deacd097412` / `e320cc9ba15c86a455036be38099429bffb828b507dbb39b09160a3bba0ecb53`; source files finite_guard.rs / lily.rs / mod.rs / hybrid.rs / benchmark script `09f7f5b68b3c521b1ab62a86c52a49cf8819abb9217002813a1410189afaf0d8` / `2587f44eef98e996040c88e28c0692e308f1386117f0951fd8f07b6a6e5b4b9e` / `6325f7cd8d163ca524c2391777983ad761a4b3e663565f666389bff4d14addc7` / `67545fe7f389772bfb1741d835ea0307d6055c670f3e9abc588919a108a2d177` / `6ddce9abd1e1bc229f52d92e3380019f658a40f0fa3bd301e02d6dec81fe5d7c`; base `f7ff311be3187de705531089b2e3081bbf001891` |

The independent NAX and GS64 summaries and their source seals are retained in `QWEN38_METAL_QUIET_RECORD.md`, `NAX_MOE_README.md`, and `GS64_README.md`. NAX source manifest/patch SHA-256: `68bd6d6c411ea26f8dcde2c3d96a8b454e5237f837d10f70bfca93003433e166` / `df506acc4f83a84d82150046e161cb5fe3c97f16e65484f0f51b9ebba2c5136a`. GS64 candidate manifest/patch SHA-256: `075bd786f4186aa921b235577d61c8fc972a2da98bb56023c9d5c4914ca1b559` / `d3af7ffaed2cf695f936b141b4651fde554a536e0774b61cc591382875ded8fe`.
