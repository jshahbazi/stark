# Hy4 Flash dense prefill and streamed expert retention

## Artifact and controls

November 25, 2026 on the 128 GiB M5 Max. The model is the verified AngelSlim
Q3_K split GGUF with the BF16 projector loaded. The revision, payload hashes,
tokenizer and patched llama.cpp reference are bound in the [native adoption
record](2026-11-25-hy4-native-q3-k.md). The source base is `6aaebc20a`, which
merges the Hy4 branch with the shared generation context sizing from main
(`08fb1723a`). Hy4 still uses its standalone fixed-capacity Forward and
NextN caches. All comparative arms hold the native and request context at
8,192 positions. The 4,096-token fixture, cap of 64, seed seven, greedy policy,
depth-two NextN verification and 4,096-row prefill chunk match the prior
[4K diagnostic](2026-11-25-hy4-prefill-decode-profile.md).

The release worker `e73c05b73ca3c5f68ddb57ee3b76386466e0c6e8d3e7ec48696004376bcb17f6`
was built with Rust 1.98.1, `--offline --locked --release -p stark-generate
--features hy4`. The private source diff SHA-256 is
`d4f221ac6d6fc6fb4e5499efcb10db42b3cd2868e1cabe943caec5bb5b2964f0`.
The aggregate-only harness SHA-256 is
`b45bcfedca7deed962aa2c5458afa4b7e9f4c60c745179c8eb519f9a6cff9f47`.
All timing arms request the same guarded 36 GB streamed arena and disable
both the stage profiler and generation prefix reuse. Each pair uses the first
and immediate second request in a fresh worker. File caches are not flushed.

## 20261125-HY4DENSE-01 · Split the prefill attention stage

Applied the dense-vs-attention attribution from Inkling to the Strata
graph. Metal stage samples cover dense attention projections including
index projections, IndexPool/scoring/selection, MLA scoring and value
reduction, and remaining attention transforms. The aggregate remains
their sum. Instrumented encoder boundaries change scheduling and are used
only for attribution.

| Stage | Encoder seconds |
| --- | ---: |
| Dense attention projections | 3.362 |
| MLA scoring and value reduction | 2.030 |
| Remaining attention work | 0.412 |
| IndexPool, index scoring and selection | 0.058 |
| Total attention | 5.863 |
| Routed experts | 4.708 |

The profile sampled all 42 blocks, matched all 64 reference output tokens
and reported 0.089 s of software expert waits. The dense projections
accounted for 57% of the attention stage. Halving their measured duration
would save at most about 1.68 s of the whole request on this geometry.
This is a ceiling calculation for the measured set of projection dispatches.
Profile receipt SHA-256
`db00a8acf33ca1cf2aadbe8e86c410fa0661a6a51e9328f802f4783e10215a2c`.

## 20261125-HY4DENSE-02 · Existing Strata dense F16 operand staging

The candidate uses Strata's existing Metal 4 half-operand pipeline at eligible
prefill shapes, including per-head K and V, still using F32 accumulation.
It applies to qualifying projections outside attention as well. Short
decode/verify windows retain GEMV and the grouped routed expert path retains
its prior F32 policy. This changes precision without converting or replacing
the original model artifact.

Both requests returned the exact 64-token reference output. The candidate
prefill rates of 290.51 and 313.33 tok/s fell within 2% of the first
control's corresponding 285.59 and 307.17 tok/s. The controls moved more
between workers. The candidate was removed. Revisit an individual matrix
only if it accounts for enough time to exceed that variation.

## 20261125-HY4DENSE-03 · Protect repeated decode expert demand

The candidate keeps two LRUs per expert size class. A second actual decode
demand promotes a resident slot to the protected queue, capped at roughly
75% of active slots. Predictions cannot promote or refresh expert priority.
Decode read-ahead uses the ordinary queue and skips when it has no eligible
ordinary victim. Required reads still respect in-flight read state and GPU
fences. The model's expert selection, file slices, precision and cache
capacity are identical.

The first bracket ran in the order below. All eight calls returned the
same 64 output tokens, 41 accepted of 43 proposals and 22 rounds. Each
worker reported 35,998,924,800 arena bytes. The host screen found no
competing heavy process or thermal warning and detected no invalid GPU
command spans.

| Arm | Prefill first / second, tok/s | Decode first / second, tok/s | Decode read first / second, GB | Target cut first / second, s |
| --- | ---: | ---: | ---: | ---: |
| Control, start | 285.59 / 307.17 | 17.16 / 22.61 | 39.17 / 27.88 | 1.660 / 1.149 |
| Dense F16 | 290.51 / 313.33 | 16.44 / 22.38 | 39.87 / 27.91 | 1.656 / 1.175 |
| Repeat-demand retention | 320.66 / 302.70 | 17.06 / 25.43 | 39.62 / 25.60 | 1.675 / 0.949 |
| Control, end | 302.91 / 272.00 | 17.69 / 23.46 | 38.47 / 27.58 | 1.667 / 1.125 |

Retention's second request improved decode throughput by 8–12% across
the two controls and reduced reported read bytes by 7–8%. Its first
request read slightly more and showed no decode gain. First and second
requests are separate comparisons. The policy can change predicted reads
as well as cache hits. Read bytes count software traffic and include
file-cache hits.

| Receipt | SHA-256 |
| --- | --- |
| Control, start | `16adbaf7ba45a1983d0032a4581ba5d7f610d27bb858c6423974890174a0de68` |
| Dense F16 | `85b7cea360858d28e23e0d01cdff97dacbb7e390602725e6504a4422db0dc3d8` |
| Retention | `70027472a6b76d21f7eca9187815548943412874e1bd832f4914e597258a2e25` |
| Control, end | `5d2e3a6cfba771b38cc7da8d134c6c7b9da850d047522f71069a9786da924abf` |

## 20261125-HY4DENSE-04 · Streamed F16 grouped expert products

Used the existing Strata M64 × N32 Metal 4 kernels with F16 operands
and F32 accumulation. The earlier half pilot combined a different tile
height, mapped banks and a scheduling change, and its host screen was
contended. This attempt isolates the available routed precision and tile
variant against the F32 M64 × N64 kernel. Geometry and arena size stay
fixed, with dense precision and the retention candidate disabled.

## 20261125-HY4DENSE-05 · Narrow decode expert read-ahead

Used Strata's existing cross-layer predictor with eight candidate experts
per row in place of sixteen. Prompt-wide preloading and required reads
remain active. This tests whether constraining speculative reads reduces
host-cut time under the same arena and demand policy.

The second bracket used release worker SHA-256
`8103acd427e43793a3268c8e0a1ac7c64042bda9dbfcbce3c5dbe21c10ec6cbc`.
Its private source diff SHA-256 was
`b0f6e24c05eadf2be9764d434ee655717f1ff8f6400978af33127ff93dcddcf0`.
The harness SHA-256 was
`e68ae8c719aeb166a77e62a48283cae3d05043aec785a80dfc286641f5d8fd73`.
PR #204 added documentation during this round. It changed no executable code.
The arms ran in the following order, with two requests per loaded worker.

| Arm | Prefill first / second, tok/s | Decode first / second, tok/s | Decode read first / second, GB | Target cut first / second, s |
| --- | ---: | ---: | ---: | ---: |
| Control | 286.41 / 305.44 | 16.93 / 23.24 | 39.28 / 27.45 | 1.641 / 1.112 |
| Routed F16 | 380.14 / 364.68 | 16.70 / 22.72 | 39.65 / 28.06 | 1.646 / 1.129 |
| Read-ahead 8 | 329.06 / 312.09 | 17.07 / 23.67 | 34.65 / 21.78 | 1.608 / 1.023 |
| Repeat-demand retention | 326.15 / 307.85 | 16.98 / 25.20 | 39.60 / 25.34 | 1.635 / 0.929 |

The host screen passed throughout and macOS reported normal memory pressure.
All eight calls agreed with the 64-token reference and accepted 41 of 43
drafts in 22 rounds. Routed F16 improved first and second request prefill
by 33% and 19% respectively. The routed precision change stays opt-in.
Warm retention improved decode by 8.5% and cut reads by 7.7%, close to
the first bracket. Retention stays opt-in for streamed experts. Narrowing
read-ahead cut software reads by 12% first and 21% second with less than
2% higher decode throughput. The switch was removed. Trial it again only
if a storage-bound workload turns that traffic reduction into wall time.

| Receipt | SHA-256 |
| --- | --- |
| Control | `a83278933e44d6daf000e11e04809a7403d969337538b98f3cef8c4c10699f9f` |
| Routed F16 | `21d7896f2f5b685c1fc16c86e2527ff5f2c7c89c83f3ba7ab12c2d54d87aa249` |
| Read-ahead 8 | `f5e0dcb970aa2939631187b43517bff8d453082d46710766f7dade030f0600fc` |
| Retention repeat | `8e01f8c0baf0bec29cea848ab161c251ead1c3eda1310c88943ef96574f6798b` |

## Final worker validation

The release worker after removing dense F16 and the narrower read-ahead
switch has SHA-256
`6d34c3f15e06afff7b45d2217c1ee482638f6c1456bad520ee3a836652a4ebe8`.
The first combined 4K attempt was rejected before starting the worker
because Cargo and rustc were active. Its aggregate receipt SHA-256 is
`95335726d05c70b66fe5561f4244402ea4f244b6c49f9056e1c8bb4935dfcde1`.
The quiet-host retry ran both options at 4,096 rows and matched all 64
reference tokens in both requests, with 41 of 43 drafts accepted each time.
The first request measured 325.63 prefill and 16.56 decode tok/s. The
second measured 368.97 prefill and 25.69 decode tok/s. This is an absolute
final-build check. The same-binary tables above isolate candidate effects.

With both retained options active and 1,024-row geometry, the same worker
passed two image cases, exact text MTP prefix restoration at 1,024 rows,
and text recovery after image requests, invalid input and cancellation.
Image generations proposed zero draft tokens and reused zero tokens.
The separate 4,096-row precision check at the same capacity compared
F16 against F32 in the final worker. It agreed on 13 of 13 teacher-forced
top-1 predictions, and both policies matched the observed next token on
6 of 13 rows. These checks cover the particular fixtures and leave
larger sampled and natural-context precision qualification open.

| Final validation artifact | SHA-256 |
| --- | --- |
| Combined 4K receipt | `980217ba172dc0d9d29e555f5096966639c9c5be333f7f6af0ffb0a4b5d4211e` |
| Image and prefix receipt | `65da2314cd32dbd0a0b7c2d101dd8990fc33c7de058ea1fdf53c1589e8a591eb` |
| Image and prefix harness | `8e330ab958947f16bf5e227919d20b781628893008994e5794a52162351d08be` |
| Natural top-1 receipt | `917136c57edc16f2f2c5b980536b38f26a7a01071cee6a20adeeaff38c598fe4` |
| Natural top-1 harness | `517c6579789c1022d3f7d2f3b27722144ca674820bc2b024d4464105643c2454` |

## Deferred work

- Price the Hy4 Forward, NextN and prompt-scoring cache allocations in
  the shared context sizing API before raising the 8,192-position native
  limit. Then validate longer prefix reuse and run the standard profile,
  whose 8,800-token prompt exceeds that limit.
- Profile packed dense projections by matrix shape and compare a narrow
  port of the measured Inkling kernel structure on the largest shape.
  The aggregate dense-half comparison does not justify converting all
  prefill projections.
- Repeat the retained precision and cache policies on longer natural
  workloads, sampled generation and different memory pressure before
  broadening the defaults or claiming a standard score.
