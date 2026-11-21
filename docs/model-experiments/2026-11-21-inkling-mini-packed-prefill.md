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
