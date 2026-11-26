# AngelSlim Hy4-flash NextN cache construction and verify attention

The [verified AngelSlim Q3_K artifact](2026-11-25-hy4-native-q3-k.md#artifact-and-runtime-identities) runs on the same M5 Max 128 GiB host with the BF16 projector loaded. In the [previous 32K run](2026-11-25-hy4-context-ladder.md#20261125-hy4ladder-02--load-time-context-sizing), NextN prompt catch-up accounted for 3.142 of 9.429 decode seconds. Catch-up remains in the decode phase for all arms.

## 20261125-HY4NEXTN-01 · Isolate work on the draft and target paths

The screening binary is built at Stark `52d6ad5242e28b06eb8a4166b9504aaf048cc447` plus source diff SHA-256 `7e990e5388835058832a32d2eb407acd0bd0ab7feb09270299629f2e2e2814e5`. Rust 1.98.1 locked offline release build SHA-256 `228053da957964029c037d903f957540a07a96b8091530ac9df1854e53e8146d`. The complete runtime shader source passed the Metal 4 syntax check. Harness SHA-256 `951d71ee3439bc0792919ee089adc58baea9c5f542ba3da7ebfb01e15a4b3409`. Bracket driver SHA-256 `06596637eee4cb58f7e091144a3ece106e94e86e21ae8d8cda6a56e9679ae56c`.

All arms request 34,816 positions, ordinary streamed arena, demand
retention, 4,096-row chunks, packed attention projections and grouped routed
F16 operands with F32 accumulation. Prefix reuse and Metal stage sampling
are off. Drafting is greedy NextN at depth two with the original model assets.
The `4K` replay requests 64 output tokens. The 16K and short coding diagnostics use
the raw-ID fixtures from the earlier ladder and request 192 output tokens.
These fixtures carry a fixed coding instruction. They do not use the pinned
chat renderer. They can screen a change on matched token IDs, but the model
card's code-generation maximum needs a coding workload rendered for the
actual model interface.

| Arm | Change and expected cost |
| --- | --- |
| Draft cache MMA | Reuse the existing F32 dense matmuls for the five NextN prompt-cache projections above eight rows. This changes their accumulation order. Single draft rows keep their existing kernels. |
| Batched draft-history writes | Keep the same eight-row operators and write their FP32 result directly into draft history in one command list per prompt chunk. It removes per-sub-batch submission and the CPU copy. |
| Resident Q8_0 embeddings | Load the verified 394,395,648-byte packed token embedding table with dense weights before context admission. Target and NextN use the existing resident reads and exact CPU row decoding. Engram remains separate. |
| Fused target MLA | Read each latent cache tile for 16 heads together and reduce online softmax partitions, keeping the exact existing sparse cells and sliding window. Query and probability operands round to F16 and scores, accumulation and output are F32. The candidate uses the existing score scratch allocation when it is large enough. |

| Arm in run order | 4K prefill tok/s | 4K decode tok/s | 16K decode tok/s | 16K catch-up s | Short coding decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control start | 481.32 | 27.14 | 23.22 | 1.522 | 25.91 |
| Draft cache MMA | 444.17 | 25.85 | 24.16 | 1.317 | 26.99 |
| Batched draft-history writes | 468.52 | 26.65 | Unmeasured | Unmeasured | Unmeasured |
| Resident Q8_0 embeddings | 456.20 | 31.52 | 28.08 | 0.207 | Unmeasured |
| Fused target MLA | 402.59 | 30.28 | 27.72 | 0.401 | 26.97 |
| Control end | 396.78 | 30.27 | 27.46 | 0.413 | 26.72 |

The closing control caught a large change in catch-up cost with the
packed embedding table still on file. Resident embeddings took
0.045 seconds at 4K against 0.083–0.306 in the controls and 0.207 at
16K against 0.413–1.522. End-to-end 16K decode improved 2–21% across
the bracket. The 4K and 16K output IDs and counts matched both controls.
Keep the resident table opt-in and charge its 394 MB to load planning.
Actual storage latency and rate gains depend on the host file cache.

The batched draft-history result fell within prefill drift. The draft
cache MMA did not beat the closing control's decode or time-to-catch-up.
Fused MLA spent 5.960 seconds verifying at 16K against 6.017–6.096 in
the controls. Its 16K output matched and the short coding output
changed. Both the useful decode rate and verification gains were small
on the measured width. All three changes stayed out of the branch.

| Receipt | SHA-256 |
| --- | --- |
| Control start | `76b79161cdcffdcf3647477c7159b2d803294d53b5d92fd47dac45128c88dc35` |
| Draft cache MMA | `414f89ef0cb41ec8e02f75aff127b6587c73a08654c82e8d2f241a93273641a4` |
| Batched draft-history writes | `e49396311eaab7780ad42d803329fb82e77743b513eb290a5ca26f2d2334ec35` |
| Resident embeddings | `129348051ff7822778fb63c35ba6bb38f174f9b02871348ca2439562d7564527` |
| Fused MLA | `8038624bf6176d291bf2ef64d2a9a47612218df111d70100d846bebe39c504f9` |
| Control end | `27da5dfd5dd5bf4225b257b9851744e13e8cc2e9268aedb8b748ec722d4fafbd` |

## 20261125-HY4NEXTN-02 · Pinned chat coding diagnostic and arena admission

The diagnostic rendered the fixed TypeScript request and archive with
the source-pinned Tencent chat template and reasoning disabled. It then
passed the resulting token IDs directly to the worker. The repeated-text
4K row retains its earlier raw fixture. The standard server renderer and
llmprobe client remain separate qualification work.

The live plan for the explicit 40 GB arena started a worker with capacity
below the requested diagnostic window of 34,816. That cell produced no
timing. The automatic 28 GB arena passed readiness and completed every
requested generation length in one resident-embedding process. Pressure
remained normal in the observed samples.

| Input | Prefill tok/s | Decode tok/s | Accepted / proposed | NextN catch-up s | Target host cuts s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 75 coding request | 55.06 | 28.70 | 116 / 149 | 0.000 | 2.773 |
| 4,096 repeated text | 461.42 | 24.06 | 41 / 43 | 0.046 | 1.221 |
| 16,384 rendered archive | 385.61 | 24.95 | 111 / 157 | 0.207 | 3.056 |
| 32,768 rendered archive | 386.44 | 23.63 | 104 / 173 | 0.430 | 3.005 |

The 4K repeated fixture matched the same 64 reference IDs. The coding
diagnostics use new token IDs and are absolute native measurements. They
need a paired reference run for an engine comparison. With the embedding
table resident, the long target verification and expert stalls deserve
priority ahead of changing the draft-cache projection arithmetic.

40 GB admission receipt SHA-256 `a45b4a80cb75ae41b9232e9817576c56f0986e24f8eb61443f57528098d2e677`.
28 GB rendered ladder receipt SHA-256 `3bfb16fac14a104ecf227a214a44a0d5c5f9e79c3fe0345e89ea646b19cf1306`.
Rendered ladder wrapper SHA-256 `3912c2f7437f786a7bc5c50810fb7750d4587f0479540ca910c8c9a6e50d8268`.
Renderer helper SHA-256 `9bb9750b3b6c191fbe4e656886ce7ea83269c5a2855d96a433f2880a9f527f70`.

## 20261125-HY4NEXTN-03 · Final resident build

The retained change uses the existing dense-buffer read of the original
Q8_0 rows. The locked offline Rust 1.98.1 release binary SHA-256 is
`5e656a4ae0a49ce030bcc19ca7f99e63f197bc4f7c2dce7101361692a98335e3`.
With both packed projections and the resident table enabled, it passed
the two projector cases, exact text prefix reuse at 1,024 positions and
recovery after invalid and cancelled image requests. Image and prefix
receipt SHA-256 `11f5aaa1cf5993f2a1424f878535ebf33d7b5fd48f19b85c9b01f06bcf5598fd`.

Three 8,192-position arena arms each ran two 4K repeated-text requests
with the projector loaded and zero reused input. Only the arena setting
changed. All six requests matched the 64 reference IDs and accepted 41
of 43 proposals. First use follows worker load. The warm row is a
second full prefill in that worker.

| Arena in run order, actual GB | Prefill first / warm tok/s | Decode first / warm tok/s | Warm expert reads GB | Warm target host cuts s |
| --- | ---: | ---: | ---: | ---: |
| 39.997 | 479.64 / 430.40 | 31.21 / 42.76 | 24.27 | 0.253 |
| 27.997 | 493.32 / 435.92 | 25.93 / 32.29 | 43.62 | 0.873 |
| 47.998 | 479.49 / 431.73 | 31.47 / 43.83 | 18.41 | 0.227 |

Relative to the 28 GB arm, 40 GB improved decode by 20% first and 32%
warm in this three-arm comparison. At 48 GB warm software reads fell
another 24% from 40 GB and throughput rose another 2.5%. The 48 GB
worker had 0.923 GB left in its context budget and could fit 12,113
tokens before the shared margin. The 8,192-position run fits. The
observed budget leaves less room for long contexts. Keep these sizes as
explicit opt-in settings and track performance and admission together.

| Final receipt | SHA-256 |
| --- | --- |
| 40 GB | `a8ad78640ed7d81f79209c1990532c43c34e2f73e188215ed7c1ae4ea14dd835` |
| 28 GB | `06d60cd5a6cdfd778a2b49bd2de417a5541c1379563193f4b55a389da19f1ddb` |
| 48 GB | `46d4cd771fce2814cfd9bc5e645c7075af18e9338de832124a9e453ab8215f66` |

## 20261125-HY4NEXTN-04 · 40 GB long-context retry

The later live budget admitted 34,816 positions with 39,996,981,248
arena bytes. The final resident build replayed the same three rendered
coding cells as the 28 GB screening build, with equal fixture hashes,
equal generated IDs and equal proposal and acceptance counts. The
projector was loaded, all 192 outputs were generated in each case,
and reuse was zero. The runs are single observations on different
builds with the same selected execution paths. Memory pressure remained
normal in the observed samples.

| Input | 28 GB decode tok/s | 40 GB decode tok/s | 40 GB prefill tok/s | Host cuts at 28 / 40 GB s |
| --- | ---: | ---: | ---: | ---: |
| 75 coding request | 28.70 | 32.97 | 58.55 | 2.773 / 2.126 |
| 16,384 rendered archive | 24.95 | 30.17 | 388.16 | 3.056 / 2.005 |
| 32,768 rendered archive | 23.63 | 26.96 | 387.11 | 3.005 / 1.963 |

The 40 GB 32K request spent 0.464 seconds in NextN catch-up.
Receipt SHA-256 `d8d10b250c904c1d560cda6a8ddc26f9d096bbde8ff710aec91f39806f921370`.

## 20261125-HY4NEXTN-05 · Remaining 4K prefill budget

With the final resident build, 40 GB and a single 4,096-row chunk, separate
stage sampling covered all 42 blocks and the same 64 reference IDs. Routed
expert preparation and compute took 2.448 s, attention projections 2.061 s,
MLA 1.880 s, dense and shared feed-forward 0.694 s, and IndexPool and
selection 0.055 s. The expert wait counter was 0.062 s. The other work and
the rest of the request account for the remainder of the 8.251 s observation.
Use these instrumented spans for attribution. Measure throughput with the
profiler off.

At 743.2 tok/s a 4,096-token request has about 5.51 s of prefill time. Wider
routed tiles and multi-query tensor MLA each address one of the three
largest spans. The observed expert waits provide little prefill headroom in
this run. For decode, prioritize protecting hot experts or removing target
host cuts without increasing the admitted physical window. Pricing the
exact prefill workspace and testing a wider grouped kernel are prepared
experiments. The published targets and the standard benchmark remain open.
Profile receipt SHA-256 `c9e37b0d8414f9a1ea99e96f9db8661719e64ab5b1ca57add10ed2ddeb288351`.

## 20261125-HY4NEXTN-06 · Wider routed tile and exact scratch estimate

At a fixed 40 GB arena and 8,192-position capacity, compared the existing
64 × 32 × 32 grouped routed kernel to the vendor's 64 × 64 × 64 kernel.
The candidate halves the K steps and output tiles and uses a larger
threadgroup tile. Both use F16 operands and F32 accumulation. The single
screening binary also moves the prefill scratch layout dry run to load
planning and uses its exact page-rounded size for the late arena guard.
That memory bookkeeping change applies to every arm in this bracket.

| Arm | First prefill tok/s | Warm prefill tok/s |
| --- | ---: | ---: |
| Control start | 488.84 | 435.13 |
| Wider routed tile | 471.49 | 443.13 |
| Control end | 446.22 | 410.63 |

All six requests matched all 64 reference IDs and accepted 41 of 43
drafts. The warm candidate was about 2% above the first control. Warm
control drift was about 6%. A separate stage sample with the larger
kernel covered all 42 blocks and measured 2.557 s in routed experts. The
earlier final build's 2.448 s is a separate profile with different clock
conditions. The wider tile has no demonstrated throughput gain and was
removed. The exact scratch estimate passed these functional runs. A
controlled admission comparison remains open, so the allocation change
also stayed out of the branch.

Screening binary SHA-256 `7119345f1c68f0941b83525a737563d4884e35a6c00233d228defb8c1a573f1a`.
Source diff SHA-256 `cb07131ecf876bb7017182fe31e575043dcadc7a3cb2f443fe383a92720af568`.
Harness SHA-256 `9a52cbd9018ad9bab9f8eb9d8df3f3aa885b2b7ad7e23be1345bafd79ec1d4f0`.

| Receipt | SHA-256 |
| --- | --- |
| Control start | `2f6e8db0eed2b174f6b3db7da3a3dd8c7f09e41b7b86e53c921eadefdce93291` |
| Wider tile | `3ca84a52faa5454b75dc1b962a586131fc69e2d3e6e18b05711937d2c02bcf31` |
| Control end | `b072e4a96c614a97dcc862ee13945bedba4c1f00540aecd88035e74a4277f678` |
| Wider tile profile | `a062e6f306872e02bd36dbc3ed11bb4ebcadfbaae7662e4bd1438a8c3c452105` |

## 20261125-HY4NEXTN-07 · Batch the resident draft's cache projections

The mapped K2 verifier keeps the target GGUF arithmetic. The candidate
uses the existing Metal 4 packed Q8_0 projections for NextN's EH, KV A,
index-key, index-value and index-gate cache rows. These projections use
F16 operands and F32 accumulation. The original GEMVs handle at most
eight rows. A complete 1,024-row chunk uses five matrix dispatches in
place of 640 GEMVs. The original prepared expert banks, resident token
embeddings, prefill options and BF16 projector are shared by both arms.

Two fresh workers from the same executable admitted 33,024 positions.
The packed arm ran first. Both ran the pinned 16K and 32K requests in
that order with K2 and no prefix reuse. All four outputs matched the
192 reference IDs. Both cases kept the same accepted and proposed
counts and target round counts. Both workers reported fair thermal
state, zero contenders and pressure level 1 with stable swap.

| Case and arm | Decode tok/s | Catch-up s | Draft s | Target verify s |
| --- | ---: | ---: | ---: | ---: |
| 16K packed | 44.97 | 0.075 | 0.591 | 3.656 |
| 16K control | 43.06 | 0.200 | 0.712 | 3.723 |
| 32K packed | 44.32 | 0.127 | 0.452 | 3.857 |
| 32K control | 41.60 | 0.419 | 0.756 | 3.834 |

The one-way pair saved 0.125 s of catch-up at 16K and 0.292 s at
32K. End-to-end decode improved 4.4% and 6.5%. At 32K the target phase
cost stayed similar and the rate remains 3.05 tok/s below the published
47.37. Retain the private candidate for order reversal and lifecycle
checks before adoption. Hold GPU embedding gathers off to keep
attribution on the five projections.

The separate mapped attribution run sampled every fourth target
verification with staged encoders and prefill profiling off. Its
owned-worker watchdog stopped it after readiness when pressure reached
level 2 and observed swap grew by 33.94 MiB. The stopped run produced no
stage or throughput result. The earlier two loads remained eligible.

The private Q5_0 draft vocabulary scout reduced the derived draft head
from 394.4 MB to 255.2 MB. Offline layout tests matched 4,101 reference
blocks. The 26.2 GB theoretical read reduction over 188 proposals is
about 48 ms at peak memory bandwidth. An additional verifier round could
erase it. The compiled option stayed off during the timed runs. Defer its
native screen and keep the target and draft vocabulary kernels unchanged.

The private source diff on `ec81193fb`, locked Rust 1.98.1 Hy4 build,
and harness have SHA-256
`abfb37b2edc035525dc680950bd0e50a352c53d49be05beae8023a81ce8b8470`,
`992b81a18e43a63e4d3b60e1a18b33a14951aabc8f572ccb75592224a475404b`,
and `b701cd263d87a96d2a431f46d38d18b206498fbbf26b1856acbf8b2c5fe3f98d`.
The pinned prior reference receipt SHA-256 is
`fa9d741e7c8b65464713449e1f8fe9677ee46b37b892c8f2715bddbdf33acd66`.

| Receipt | SHA-256 |
| --- | --- |
| Packed projections | `5cf280c1f993a9de8d663b6bb9f9bf9a717e5c8776f248673c0fb0ba5aeedf1d` |
| Control | `9b7b2c06f552e68e45ed74b8fef9ca420ba032d6ba9d908e3861c1d5c1143645` |
| Sampled verifier pressure stop | `972a8506e1c58f84fadc6ea686b22da38276e32c16301cc6eb72d068e4da33bb` |

### Repeat after the host restart and sample 16K verification

The first reverse-order run reached readiness before the host rebooted.
Its receipt has no generation or terminal result. The retry used the
same executable, references, 33,024-position cap and 4,096-row chunks.
The control reported pressure level 2 after readiness and the watchdog
stopped it after a 133.94 MiB swap increase, before any timed result.
The packed worker completed both requests at 44.56 and 43.36 tok/s.
Catch-up was 0.075 and 0.127 s. Both outputs and all draft work matched
the first pair. Its pressure and swap held at level 1 and 125.94 MiB.
The host reported nominal thermal state and zero contenders. The retry
confirms the candidate's catch-up cost and yields no reversed rate
comparison. Keep the packed path private.

A 17,408-position load sampled 20 of 79 target rounds at 16K. It
matched the 192 reference IDs and stayed at pressure level 1 with
46.06 MiB additional observed swap. The samples attributed 0.290 s to
routed products, including the GPU plan and combine, 0.242 s to dense
attention projections, 0.095 s to remaining dense work including the
head, 0.077 s to MLA, 0.065 s to attention elementwise work, 0.038 s to
IndexPool, and 0.031 s to the router. Encoder boundaries affect
scheduling. These totals price stages for investigation.

| Receipt | SHA-256 |
| --- | --- |
| Interrupted reversal | `642dd276fcb40ed1675cdc32f3d0695f80ff945afb1312bc7fd423532c1ff9d6` |
| Retried reverse control pressure stop | `52ca71b4da7ff407529c517062cd9eff91d50f5f21a789d338404bedcd37052c` |
| Retried packed requests | `2905ecac95f8d861f2afe0949f3d82335432e99e212f9fbe95c809fc555b2e44` |
| 16K mapped stage sample | `8617f834e429d77bbf02eb36673c68f1d49b51e67874adf1431820359910a071` |

### Reduce long-window prefill workspace and test paired gate rows

The [HY4FRAG-27 executable](2026-11-25-hy4-grouped-fragments.md#20261125-hy4frag-27--screen-smaller-routed-staging-and-group-local-dense-packing)
used 1,024-row prefill chunks and a 17,408-position context cap.
All completed K2 requests matched the pinned 192 output IDs and the
earlier draft counts. Observed pressure stayed at level 1 with no
swap growth on the three completed workers.

| Arm in run order | Decode tok/s | Catch-up s | Target verify s | Receipt SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Control, compiler detected | No timing | | | `f72bb38be977db5c8fd3bf601fd3369025e9ac8bf515318ff3ab2c8da56712d6` |
| Packed cache and paired gate output rows | 28.04 | 0.066 | 6.346 | `e84ae21916c8496180d85bb6dedb64b0b7190ed24bf7ee7f0ce91d26d8611dc7` |
| Packed cache | 45.25 | 0.066 | 3.760 | `e39b9024a0ca3eb0f346449b414156cbcb2b26ac3bfe158914c1c52eccf4bc56` |
| Control retry | 42.97 | 0.214 | 3.826 | `8ae8e62c335d85e0fa35add36d83ddbff5e0dd3b928ef8618e5b28af436627ce` |

The rejected first control detected a separate Metal compiler at
admission. Both the packed cache and retried control reported nominal
thermal state and zero contenders. Draft catch-up fell by 0.148 s, a
second scoped pair with the same direction as the 4,096-row chunk
comparison. The packed cache rate still trails 49.47 at 16K.

Pairing adjacent output rows of Q3_K and Q4_K gate/up products keeps
the eight-lane FMA and reduction order, and halves repeated logical
activation loads for three verifier rows. This arm moved from fair
to nominal thermal state. Its draft phase stayed within 6 ms of the
packed-only arm and target verification cost rose by 2.586 s. Remove
the pairing and keep the existing gate geometry.
