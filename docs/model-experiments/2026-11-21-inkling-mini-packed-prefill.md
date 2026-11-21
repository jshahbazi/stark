# 20261121-INKPREFILL-06 · Inkling 2 Mini packed prefill

| Field | Recorded value |
| --- | --- |
| Artifact | Original vision-language target at revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, shard SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1` |
| Incumbent | `d8963ab83ab3b9bc83098d113d9c2ea720701014`, 128-row commands |
| Host | 128 GiB Apple M5 Max, macOS 26.7.2, locked Rust 1.98.1 release |
| Native control | 4,096-token pinned fixture at context 4,100, private bundle SHA-256 `74d6457708d31fe3ba6aeeab41b836d6ac7cd825cb3837fc3ddacc64e0c19719` |
| Acceptance | Exact committed KV and convolution state, BF16 target taps, four F32 continuation rows, snapshot replay and complete worker output hashes |
| Predecessor | [4K prefill budget](2026-11-21-inkling-mini-prefill-budget.md) |

## Packed activations

The prefill graph stores BF16 and F32 activations by token row and lets consecutive kernels read the same buffers. Residual mixes, RMS, controller expansion, dense projections, history convolution, head normalization and cache commit run across each chunk. The attention stage reads new keys and values from the chunk until all queries finish, then commits the latest keys and values per cache slot. This preserves the previous ring slots on a chunk that crosses local wrap. The decode path still uses the single-row graph.

At 128 rows the packed graph went from 573.022 to 305.975 ms over eight alternating pairs. All eight matched state, taps, four continuation rows and snapshot replay. The 512-row exact graph ingested the 4K fixture in 11.095 s against 20.669 s in one pair, or **369.18 tok/s**, with all the same checks passing. The packed workspace added 72.3 MiB at 128 rows and 252.3 MiB at 512.

The separate instrumented 512-row profile encoded 41,688 dispatches over eight commands and reproduced its uninstrumented continuation logits. The preceding 128-row profile had 5,131,776 dispatches. Command GPU time moved from 22.504 s in that profile to 11.861 s in the packed exact run. Those profiles were taken in separate sessions. The exact packed stages spent 4.087 s in NVFP4 experts, 2.465 s in MXFP8 experts, 2.685 s in dense attention projections, 2.337 s in relative/cache attention, 0.255 s in embedding and preparation and 0.032 s in command gaps.

## Kernel screens

For the 512-token screens, the baseline and candidate used the same loaded checkpoint over eight alternating windows. Ratios are the median of the per-pair wall ratios.

| Change | Ratio | Result |
| --- | ---: | --- |
| Packed matrix dense | 0.794 | Changed state, taps and continuation in eight pairs. |
| Query-tiled matrix attention | 0.946 | Changed state, taps and continuation in eight pairs. |
| Tiled matrix MXFP8 | 0.785 | Changed state, taps and continuation in eight pairs. |
| Fused exact attention reduction | 0.997 | All parity checks passed. Rough tie. |
| Share dense weights across 16 rows | 1.053 | All parity checks passed. Slower. |
| Matrix dense with conservative BF16 intervals and GPU replay | 1.177 | All parity checks passed. Slower. |
| Share MXFP8 weights across eight routes | 0.912 | All parity checks passed. Selected for the first worker comparison. |
| Guarded tiled matrix MXFP8 against eight-route MXFP8 | 1.058 | All parity checks passed. Slower. |
| Guarded 64-K NVFP4 against the 32-K kernel with eight-route MXFP8 active | 1.096 | All parity checks passed. Slower. |
| Share the same K/V loads across four adjacent exact queries | 1.003 | All parity checks passed. Rough tie. |
| Guarded dense with tighter bounds, worklist compaction and two dots per SIMD group | 1.084 | All parity checks passed. Slower. |
| Use actual decoded-weight L1 norms in the 32-K NVFP4 guard | 0.963 | All parity checks passed. Below the practical gain threshold. |
| NVFP4 activation with a three-comparison FP4 midpoint search | 0.993 | All parity checks passed. Rough tie. |
| Barrier-free NVFP4 matrix fragments in SIMD registers | 1.159 | All parity checks passed. Slower. |

The combined matrix path ingested the full fixture in 5.309 s, or **771.47 tok/s**, in one pair. It changed committed state, taps and continuation logits. Its 5.638 s instrumented command time included 4.066 s NVFP4, 0.237 s MXFP8, 0.681 s dense attention projections and 0.386 s relative/cache attention. The path is diagnostic only.

In a separate run the matrix path reached 676.79 tok/s. It agreed with three of four pinned Ollama greedy continuation decisions. The incumbent agreed with one. Relative L2 was 29.91%, 18.92%, 14.26% and 11.18% for the matrix path and 35.21%, 23.63%, 13.18% and 4.16% for the incumbent. These four teacher rows support diagnosis. The two full-model paths still have substantial reference error. The guarded matrix experiments also rely on an assumption about intermediate FP32 accumulation. Apple's examples specify a float destination tensor. [Metal Performance Primitives programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)

At 4K, 1,024-row commands took 11.125 s against 10.918 s with exact output and added 492.4 MiB of workspace. The shared K/V and tighter NVFP4 guard together took 11.043 s against 11.369 s in a single exact full-model pair. The register-only NVFP4 kernel took 12.056 s against 10.212 s in the exact full-model pair. All three kept their short-screen dispositions.

The eight-route MXFP8 follow-up used a separate NVFP4 stage profile. Its 9.147 s instrumented 4K budget included 2.994 s for NVFP4 matrix products, 0.273 s for their router and bucket stages, 0.069 s for activation quantization and ingress, 0.079 s for epilogues, 1.282 s for MXFP8, 2.223 s for dense attention projections and 1.971 s for relative/cache attention. Uninstrumented target ingestion took 8.477 s in that session and reproduced the instrumented continuation logits. The result puts a tight speed ceiling on further activation quantization work.

## Incremental worker candidate

The exact 512-row worker with eight-route MXFP8 was screened with vision and draft weights loaded, zero reused tokens, draft execution off, 4,096 input tokens, 4,352 context slots and 256 emitted tokens per request. The worker order was incumbent greedy, candidate greedy, candidate sampled, incumbent sampled. Both output hashes matched.

| Request | Incumbent prefill | Candidate prefill | Incumbent decode | Candidate decode |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 199.76 tok/s | 473.05 tok/s | 33.08 tok/s | 34.85 tok/s |
| Seeded sampled | 180.87 tok/s | 406.67 tok/s | 28.96 tok/s | 29.87 tok/s |

The greedy prefill took 8.659 s and the sampled prefill took 10.072 s. These were single pairs with distinct requests. The receipt SHA-256 is `8f2f3bcba4ba69efb50b548cedfc9d60bc0a862da99b002e20a6cd63492522da`. The candidate executable SHA-256 is `a6591310e7207c0a191aa77402e5e334d0ad491aaefd10212d8aef5552cfdf28`.

The worker restored 2,048 shared tokens and ran a changed 471-token tail across the packed graph. Prefill went from 5.063 s cold to 1.091 s with the restored prefix and reproduced the 24-token output hash. Cancellation during the following prompt cleared the saved prefix. The cold retry reused zero tokens and matched the same hash. The desktop-facing smoke also exercised image readiness, a visible image answer, greedy and sampled draft checks, streaming and a tool-result round trip.

## Selected default and release controls

The branch uses packed text prefill for 16 or more rows on the M5 Max when drafting is off. Its per-command bound is 512 rows. Images, active drafting and shorter tails continue through the existing per-row-input graph. The worker uses the same 1,024-token snapshot alignment and retains the target KV and all convolution histories in the existing prefix snapshot. Other hardware retains its existing memory reserve and scheduling.

The final example won eight of eight held-out 159-token pairs with a median wall ratio of 0.494. The 4K pair took 12.117 s against 22.693 s in that session. Every pair reproduced committed state, BF16 taps, four F32 continuation rows and snapshot replay. The short and 4K connected slice aggregate hashes matched their pinned predecessor exactly. The synthetic 4K decode control measured 28.419 ms median across 32 steps, one completion and 1,883 dispatches per step. These results bind to example SHA-256 `edca711ccbf3ea4ac81f487d2af8fd16d045be693aca5273c8a82f960e0db117`. The 159-token, 4K and decode receipts are `f0d2368e70af2b9a72b9b880db846ed085ba001c5a1f78ec3f29c0060501bf68`, `436f0214039f103cd76676d4b68e2b7c1eddc5fa3fc9540610a3c542dcb14f28` and `d7204ad52133b204c03d6e0e5517ed82b043c95758c8c182dc6ade047231d10d`. [Pinned short and 4K slice results](../benchmarks/2026-11-21-inkling-mini-native-worker.md)

The final worker restored the same 2,048-token prefix and matched the continued output, reducing prefill for 2,519 tokens from 7.707 to 1.460 s. The retry after cancellation matched and reused zero rows. Image, draft and API smoke checks passed again. Final prefix and desktop receipts are `4b5a6cfe5dd99cf8350ac757d44aa18692b88ece5890e8d20db7e919f9452cc1` and `3a80387c428e2436bbd18ba04d43ac057fffa65fb7910f0a11d956462afb7e63`. The locked worker suite passed 157 tests with five private-artifact cases ignored. Repository policy and formatting checks passed. [Final cold 4K worker timings](../benchmarks/2026-11-21-inkling-mini-native-worker.md#packed-512-row-prefill)

## Evidence identities

The first exact and matrix screens used example SHA-256 `9aaaff545ce629412c7b8ba4c00d43053b3d3ad78e85580a8dcdbbc0ca9d45a6`. The corresponding 4K receipts are `e855346a164981e5b822e1645f2c234aa8436c40d551ed5fc980ef6511f4059e` and `4453758f904c6929871f7b1a9297c35fd4fbf9f0cc9a7b9c541268b6d3cc1376`. The profile executable SHA-256 is `8d02b32878354d038a5d126aad9984eb1400fa8e4e40f2827f6c706fdef7235e`. The first fused, guarded dense and guarded expert screening executables have hashes `d477b7a91459cc7d1f4e7f65959256e792e597cb8b5f99026f8f791e8c79aef7`, `a0813572b01f0de32873e1b3e4d068316c6e32c99d48c8bf70a5deeb48607287` and `75c869e502f1371a16c233f33340b8279955ce2fab6a51de29fab58acf120855`. [Experiment ledger](README.md)

The later screening examples have SHA-256 `7710b6ea14a945d4dc1f4af27a5d49528b1375f2829b4af211729e68ef873121`, `9fef60c6e45b686bccad7f13d300e6ea10841f15f40819679a86b22b9f59f4b7` and `9e02e209837fdd781e68f8d226e564b07a188a471ca8af7d41335ff600acd49c`. The finer stage profile receipt is `def9881f17a04ef1a563300a9672ea6d66190869c0720dde193f15fcb83a9ab9`. The final matrix-source comparison receipt is `35a9e836cf692cef707e880e25083ef83bb9f5f4f98de9c35da9d5bc9fbd3fec`. The complete probe diff and added shader archive remain private with hashes `f6984320f5d37816e98b163fbf53777f2f7fd5a0237d5a1c09056acea24902c5` and `f22a1847ee56ac411250a52d2749c0ed39fc30cdbb2cefc57f9d1193cf83b2bd`. The first live cache and desktop receipts are `78f3f6e8a83044a414cd36d5cbf20c036dbafe5a47f225b9a649f29abbf953db` and `ade0d7a3cc28e417d033e42f714131767ec1ad077702d2c292c8bd376f575d80`.

## 20261121-INKPREFILL-07 · Source-precision holdouts

Three additional 4K inputs were captured with the same original checkpoint and pinned Ollama runner, eight 512-row prefill calls and four singleton teacher forwards per input. The sample set includes the first sampled worker benchmark input and two additional windows. Each Stark comparison ran one same-session pair at context 4,100. The candidate had to retain the incumbent's greedy teacher agreement on each input and keep mean reference-relative logit L2 error within two percentage points of the incumbent. A teacher row with more than a five-point error increase also blocked promotion. The paired preparation and ingestion rates here are target diagnostics.

The dense matrix arm uses the previously captured MLX NAX fragments. The dense and local arm also uses the source reverse 128-key kernel, BF16 relative table and rounded unnormalized probabilities, with a gather from the packed chunk and its previous ring. The router arm additionally uses the split-K source TF32 router and the source-shaped route arithmetic. Global attention and expert matrix products use the native incumbent math in these three arms.

| Arm | 4K diagnostic prefill | Pinned top-token matches over 12 teacher rows | Largest increase in row logit L2 |
| --- | ---: | ---: | ---: |
| Incumbent exact packed | 431–482 tok/s | 9 | Reference for the change |
| Source-shaped dense | 500–523 tok/s | 8 | 67.55 percentage points |
| Source-shaped dense and local | 647–650 tok/s | 7 | 66.54 percentage points |
| Source-shaped dense, local and router | 638–653 tok/s | 7 | 49.35 percentage points |

The source-shaped dense arm reduced mean logit error from 37.96% to 11.61% on one held-out input and increased a different teacher row from 15.99% to 83.54%. Matching the local kernel reduced that high error to 82.53%. The direct-graph speed cleared 500 tok/s. All three arms failed the source numerical gate and remain diagnostics.

The separate exact attention probe used one exponential per key, taking advantage of the unit weighting on the larger logit and retaining the original arithmetic for nonfinite comparisons. It passed all eight short numerical pairs and the 4K sampled fixture. Median short wall ratio was 1.006 and its full pair ratio was 1.023. The probe was retired. Its source patch and 4K receipt are `5e0bfb8dbc050caae32bbb56ee3882f4cfc809a92d4c2f9f162bd25f5e7097c3` and `7d0bbb8ff537947fe29f8d59d630a858589cda32435b4787baa7142e8c5cb1a0`.


## 20261121-INKPREFILL-08 · Source arithmetic and embedded library

The packed source graph uses BF16 NAX for dense projections and local relative attention with BF16 probabilities, F32 global attention, and FP32 split-K routing. It matches the source’s stable gate math and the MLX LogAddExp and softmax sequence. Native grouped MXFP8 assigns eight contiguous codes per lane for gate/up and four for down, multiplies each partial sum by the E8M0 scale, and keeps the BF16 projection boundaries. Dynamic convolution rounds the product and running sum separately in F32, as in the pinned capture. The original exact packed arm keeps its previous precision policy.

Trace-driven corrections were measured in order. The NVFP4 epilogue cleared 8,780 differing expert output words on the first 512-row visit. Matching LogAddExp cleared the first failing route weight in visit 4 and moved the first connected discrepancy to MXFP8 at visit 44. Switching MXFP8 to the source router reduced its expert mismatch from 2,280 to 342 BF16 words. The MXFP8 lane layout cleared all 342, and all 48 residual visits then matched in the captured first chunk. A separate input’s second chunk still differed by one BF16 value-convolution word at visit 11. Separate FP32 multiply and add cleared it. That input’s captured chunks at offsets 512, 2,048 and 3,584 each matched all 48 residual visits after the fix. The independent CPU convolution check reproduced the one-word difference and the repair for rows using only current-chunk history.

The 512-row oracle deliberately uses eight explicit calls and then feeds four singleton teacher tokens at positions 4,096–4,099. The pinned Ollama generation scheduler uses prefill widths 2,048, 2,044 and 3 on an uncached 4,096-token text request. It then forwards the last prompt token at position 4,095 to seed generation. The forced-width capture qualifies the corresponding boundaries. The stock-schedule check uses its own capture of the seed and subsequent teacher rows.

The final packed prefill candidate before singleton fixes cleared the held-out top-token and mean-error screens on five inputs, with 17 of 20 teacher top tokens matching the source versus 14 for the incumbent. Two teacher rows still exceeded the per-row guard after entering singleton continuation.

| Input | Packed source prefill | Incumbent → candidate mean logit L2 | Pinned top-token matches | Largest increase in row logit L2 |
| --- | ---: | ---: | ---: | ---: |
| Greedy 1 | 701.5 tok/s | 37.96% → 12.46% | 3/4 → 3/4 | 5.19 pp |
| Greedy 2 | 609.1 tok/s | 18.48% → 15.28% | 3/4 → 4/4 | 7.97 pp |
| Sampled 1 | 618.7 tok/s | 20.56% → 11.61% | 2/4 → 3/4 | 1.72 pp |
| Sampled 2 | 659.7 tok/s | 29.76% → 13.68% | 4/4 → 4/4 | −0.84 pp |
| Sampled 3 | 616.9 tok/s | 23.32% → 8.94% | 2/4 → 3/4 | −4.14 pp |

The four stock Metal kernels build from 22 unmodified headers at the pinned MLX revision and link into the Stark worker. The compiled library preserves the connected trace results of the released Ollama library. The packed source worker scope is text with drafting off at context up to 4,352 and packed chunks larger than 64. The same worker keeps the incumbent path for short chunks, longer contexts and image requests. The candidate measured 517.6 MiB of Metal scratch across both prefill workspaces, within its 576 MiB reserve.

The worker benchmark below uses the original vision-language checkpoint with vision and DSpark resident, drafting off, 4,096 input tokens at context 4,352, zero reused input, a ten-second post-readiness pause and exactly 256 output tokens. Each cell has one valid pair in each session. Both sessions had different generated hashes across the two precision policies.

| Session | Greedy incumbent → candidate | Seeded sampled incumbent → candidate |
| --- | ---: | ---: |
| Source kernel probe | 389.71 → 613.39 tok/s | 437.41 → 626.19 tok/s |
| Build with embedded library and conv fix | 487.95 → 772.04 tok/s | 465.50 → 771.67 tok/s |

The embedded candidate delivered 35.65 useful decode tok/s greedy and 33.72 sampled. Separate prefix replay restored 2,048 tokens and processed a changed 471-token tail. Prefill fell from 3.226 s cold to 0.656 s with reuse and reproduced all 24 output tokens. Cancellation cleared the retained state and a new cold request reproduced the same output.

The packed trace executable SHA-256 is `ded04183d9b1747d276a5bdb902b8c56178e0c44ddf2db6d81ddc666f7455c54`. The last-chunk trace aggregate is `6810d99cb31e433761fdc4b93f4c5776ae667b944247b1074a24c7382c6bfd8a`. The embedded candidate worker SHA-256 is `f84fda3dfb3632faa13d98ea7cd1e44792141e3daa9f6f8b2e93554739acea19`. Its worker timing and replay aggregates are `5cf6b91b62f05512ab934fcb83347eed8e1cd9cf61722230199931abf86854bc` and `3c8cac33aa1680c278105e389c43e2fc103ed0c805069ba57b1529355e8a76f9`. The first reference-library worker aggregate is `4b302ec5165925d65a4224ecbfa88e8a4b0624eaf4a6c37fd7014ca8d5436a74`.


## 20261121-INKPREFILL-09 · Source-aligned worker

The source-aligned worker batches 2,048 bulk rows and preserves a four-token final suffix, including the generation seed. A cold 4,096-token request runs 2,048, 2,044 and 3 prefill rows before the seed at position 4,095. The three tail rows use width-stable arithmetic in one Stark command. The source reference uses one three-row forward before the singleton seed.

The worker inherits the source’s distinct BF16 Thin projection tilings for 2,048 and shorter bulk batches and chooses split-K routing using the request width. Singleton continuation uses the source router, expert gate, relative table, K/V normalization and convolution. The four final MXFP8 visits use the contiguous per-lane QMV layout. This fixed a single differing BF16 word at visit 46 of one held-out generation seed. The complete stock-schedule test passed all 16 F32 logit vectors and all 48 BF16 visit outputs per vector across two greedy and two sampled inputs. The measured native target ingestion time across those captures was 5.077–5.239 s. Snapshot replay was exact on all inputs.

The source path requires an M5 Max, text-only input with at least 512 tokens, context at most 4,352 and draft execution off. The loaded vision and draft payloads remain available. The source workspace and 128-row workspace add 1,245.0 MiB, covered by a 1,344 MiB reserve. The worker admits the smaller 384 MiB incumbent workspace when the source reserve does not fit. If source setup fails, it retries the incumbent. `STARK_INKLING_SOURCE_PREFILL=0` selects the incumbent at startup. Image input, draft-enabled generation, short text and larger contexts retain the earlier paths.

Source-mode prefix retention saves a complete 2,048-row bulk boundary. The incoming request must still reach that boundary before its four-token suffix, under the same scope, context and draft mode. This preserves the later bulk widths and the final tail on a restored request. [Capture instructions](../archive/experiments/2026-11-21-inkling-oracle/README.md#stock-schedule-4k-text-capture)


The singleton router fuses selection, pinned LogAddExp, negation, softmax and scaling. The four removed dispatches per visit leave all 16 stock-schedule logit vectors and their BF16 visit residuals exact. The final M5 Max worker won all eight cold 4K pairs per mode, with 683.34 tok/s greedy and 685.31 tok/s seeded sampled at the medians. This was 1.776× and 1.780× the previous worker on a paired basis. The corresponding requests finished about 23% sooner, with about 6% lower useful decode rate inside the absolute decode budget. [Final worker benchmark](../benchmarks/2026-11-21-inkling-mini-native-worker.md#source-aligned-m5-max-prefill)

The pinned stock-schedule runner SHA-256 is `64293fa71651074e08703bdcd33d66c870b1ea23967f04f9da17bd6f095693b5`. Its first attempt reached all scheduled prefill rows and then failed while capturing the nonaligned ring view. The corrected fixture slices both tensor dimensions. The probe executable with fused routing SHA-256 is `df8890c779bf0a9854f25a7417775953dfca943fe6e36aef6e8c141f4a5204f6`. The four stock-schedule replay aggregates are `ea964e1c4694e5a6dcc793d9b4607fb17baf6d9082331db69bd155c013d9a29b`, `5eea427ff921422a9350be8e3858cc50c1c1ca4c01e97b0e0ae73066d2a8edb5`, `49bbf73fdfd3aca76cd90c261c8762d72116f9a051fbd80464242b264ebb126f` and `ebd5ad1e77fd9120f8481b4965a3d1cd0869035a3b183c80af68ba452f193287`.


The final worker restored 2,048 tokens into a changed 471-token suffix and reproduced all 24 cold output IDs. Prefill in that replay went from 4.139 s to 0.891 s. Shortening a request from 2,052 to 2,051 tokens or from 4,100 to 4,099 correctly rejected the saved 2,048- or 4,096-row boundary, with exact cold output. Cancellation cleared the saved state. The prefix-control aggregate is `7f1db5bbb61ed489a7ef61650b3ee11a455b07b6df50f0d657084dfc965675b8`. The primary greedy and seeded output hashes matched both earlier source-schedule repeats after the routing fusion.


The final worker reproduced the incumbent hashes for short text, a 4,608-context request and the explicit source-prefill kill switch. The desktop API checks passed streaming, the image color control, greedy and seeded draft-on/off output equality, and a tool call and result round trip. The fallback and desktop aggregates are `ed480a82aa33d016bf93566547abf1a31571db20439c39cb9b706b2c0ec45c3b` and `c72a4e1633cb014b9404eb6461e30d2fcdb854a68a11a74e669638461c97193f`. The locked worker suite passed 157 tests with five ignored. Repository policy and Rust formatting checks passed.


The final default trace build reproduced all 16 stock-schedule vectors, all per-visit residuals and all four snapshot replays. It also reproduced the archived 4- and 4,096-token connected slice hashes. The synthetic incumbent decode control reran at 28.030 ms median with one command and one wait per step. The final trace executable SHA-256 is `b6c0c2d65101e39198b3680fe5882fe91092e950dbcfc38fdf42ff5588ffa1f4` and the synthetic control receipt is `e39a5628249059415dad50786329468a16d966ee71a96f69b6b30181d210d01e`. The four final stock-schedule receipts are `8290935897a14110c6e1338fcc9ee0fb9d2f634dac548d5df64635c744d56e90`, `1c2f632175ba1a94fa040fcd4309164241ad25551af4e190d8a79353e84b2d0b`, `34df1da254bd771c42dbaa59d08837e5586128225c510325bc3f1fb62e36717e` and `139348ed520309d03e41721f5da6a5d6f314c9b8931dad4417d11cc69b17844d`.
