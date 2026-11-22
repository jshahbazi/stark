# 20261121-INKDECODE-01–29 · Inkling 2 Mini exact decode toward 60 tok/s

## Scope and budget

The target is sustained useful decode of at least 60 tok/s on the original Inkling 2 Mini checkpoint and 128 GiB M5 Max at cold 4K. The worker uses vision and DSpark resident, context 4,352, 256 output tokens and zero input reuse. Keep the promoted prefill policy and match target arithmetic. Include both greedy and seeded sampled worker measurements before claiming the target. A diagnostic command has a different boundary from useful worker decode.

The starting worker at `a370a786` measured 33.78 and 32.12 useful decode tok/s, or roughly 29.60 and 31.13 ms per interval. The 60 tok/s budget is 16.67 ms per interval. Its same-cache profile attributed about 12 ms to NVFP4 products, 5–6 ms to dense attention projections and about 5 ms to router projection and route selection. One draft with a 5.2 ms startup cost leaves a maximum of 28.13 ms for verification, rollback and host work at 60 tok/s even if both rows are useful. The strongest route combines a cheaper singleton with weight reuse across multiple verifier rows. [Default router baseline](2026-11-21-inkling-mini-decode-default.md) · [Earlier speculative design](../INKLING_2_MINI_PERFORMANCE_FOLLOWUP.md#largest-credible-gain-is-a-verifier-that-can-skip-target-work)

## 20261121-INKDECODE-01 · Exact layer-major verifier screen

Reuse the per-row target activations and shared projections in `PrefillLanes`. Read all query rows from one padded vocabulary projection. Run the candidate from the same restored short or 4K cache as the serial teacher sequence. Compare each F32 logit vector and the committed KV and convolution state after the complete block. The diagnostic restores its prefix outside the timed command and tests the fully committed block. A rejected-suffix transaction and worker integration are separate work.

Widths 2, 3, 4, 6 and 8 reproduced every F32 logit vector and committed cache hash on one short and two stock 4K prefixes. Timing medians use four comparisons at each width. The plain layer-major decoder shares target weights where the existing singleton arithmetic permits it. The grouped trial also reuses exact source expert routing and products. It caps its grid at the number of routed rows because each live bucket holds at least one route.

| Rows | Serial command median range, 4K | Layer-major singleton expert median range, 4K | Grouped expert median range, 4K |
| --- | ---: | ---: | ---: |
| 2 | 57.77–58.19 ms | 48.22–48.76 ms | 44.42–44.97 ms |
| 3 | 86.49–86.69 ms | 66.84–66.85 ms | 50.22–51.19 ms |
| 4 | 115.76–116.31 ms | 85.30–85.85 ms | 55.26–59.50 ms |
| 6 | 173.75–174.06 ms | 124.47–125.42 ms | 68.68–76.76 ms |
| 8 | 230.81–231.64 ms | 161.81–162.38 ms | 80.10–90.11 ms |

The grouped arm reproduced every checked row and committed-state hash at the same widths. Its benefit already grows with block width. A four-row round still needs substantially cheaper expert work or stronger proposal acceptance to reach the goal. The initial four-row profiler exhausted its encoder timestamp slots. The 8,192-slot allocation also failed on this device. Folding attention and expert subphases into stage totals fit the existing 2,048 samples. The failed profile runs yielded no reported timing.

## 20261121-INKDECODE-02 · Native multi-mask DSpark costs

Preserve the full eight-row bidirectional draft block. Share one vocabulary projection across the requested mask rows, and carry the BF16 Markov state and preceding proposal through the recurrent head. A separate small recurrent scratch keeps the shared vocabulary logits live. On all 96 greedy diagnostic rounds, the first mask matched the existing one-proposal implementation and the depth-one and depth-three proposals matched the corresponding prefixes of depth seven. The test uses the existing draft-on target precision policy. The two 4K labels identify prompt sources and both continuations use greedy target draws.

| Starting state | Accepted by proposal position, out of 32 rounds | Median proposal ms at depth 1 / 3 / 7 | Verify + finish budget at 60, depth 3 |
| --- | --- | --- | ---: |
| Four-token prefix | 26, 23, 18, 11, 6, 2, 0 | 4.50 / 5.21 / 6.66 | 46.35 ms |
| Stock 4K input A | 11, 0, 0, 0, 0, 0, 0 | 4.90 / 5.61 / 7.07 | 16.79 ms |
| Stock 4K input B | 21, 11, 6, 2, 0, 0, 0 | 5.35 / 6.10 / 7.63 | 30.36 ms |

Each count represents an accepted prefix through that position. The budget divides the expected committed tokens per round by 60 and subtracts measured draft time. That time must cover verification, rollback and host work. One stock input had no second-position acceptance. Depth alone cannot close the gap, and pooling the two inputs would hide that failure. Future work needs a faster ordinary token on hard inputs and a verifier that adapts to useful proposal depth.

## 20261121-INKDECODE-03 · Exact vector NVFP4 singleton

Group the 16 original FP4 block products into four vector dots. All 16 activation values use the same decoded E4M3 scale and all FP4 values are half-integers. Factoring out their common power of two bounds the remaining integer coefficient by `15 × 16 × 12² = 34,560`, within exact FP32 integer range. This permits the inner regrouping. The scaled per-lane FMA recurrence, SIMD reduction, BF16 round and nonlinearities keep their existing order.

The vector arm reproduced every checked F32 logit and committed cache hash on one short and both 4K inputs. The 4K uninstrumented singleton medians moved from 29.27–29.28 to 24.29–24.33 ms. The instrumented NVFP4 products fell from about 10.04 to 4.95 ms. The four-row singleton-expert verifier also fell from about 85 ms to 65 ms. The vector path has a worker selector, `STARK_INKLING_EXPERIMENT_NVFP4_VECTOR_BLOCK=1`, inside established M5 Max source scope.

### Worker measurements

Both arms ran one sealed Rust 1.98.1 build with identical promoted prefill and model artifacts. Only the candidate enabled the vector block. The launch environment had no other Inkling flags. Eight alternating pairs per mode passed the 4,096-input, 256-output and zero-reuse checks. Every output hash matched. The paired gain is the median of the per-pair rate ratios.

| Sampling | Ordinary singleton | Vector singleton | Paired gain | Vector range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 32.32 tok/s | **38.66 tok/s** | +20.74% | 37.13–40.49 | 8/8 |
| Seeded sampled | 31.04 tok/s | **36.75 tok/s** | +18.46% | 35.95–37.78 | 8/8 |

Paired prefill movements were -1.24% greedy and -0.12% sampled. The workers accounted for 99.86% of sampled GPU counter increments over 219 intervals. The host sampled nominal and fair thermal states. Pageouts moved by +117 pages and swapouts by +0. These are shared-host paired measurements.

With the vector flag set, short and connected 4K aggregates repeated exactly. Four pinned 4K captures matched 16 F32 vectors, 768 visit residual blocks, each snapshot replay, and the first continuation in the stock profiler. The sealed worker matched 16-, 47- and 64-row packed tails, reused a 2,048-token prefix with exact changed-tail output, rejected both invalid packed boundaries and cleared the cache on cancellation.

The final build isolated the multi-mask draft scratch and shaders in the diagnostic. A fresh short and 4K check reproduced the scored vector outputs and cache hashes. One final 4K worker pair per sampling mode reproduced the scored output hashes. The throughput table comes from the original matched scoring window.

## 20261121-INKDECODE-04 · Parallel exact expert selection

In the source singleton, one SIMD group scores four experts per lane. At each of the eight selection steps, it reduces the maximum score and then the lowest expert ID at that score. It masks the selected score and retains the prior MLX normalization order. Non-finite score cases that contain a NaN use the prior ordered scan. The experiment is opt-in with `STARK_INKLING_EXPERIMENT_PARALLEL_ROUTE=1` in the existing source text scope. The original one-row projection runs in both arms.

Against the preceding shader, 99 synthetic fixtures matched the eight selected IDs and eight F32 weights exactly, including ties, NaN and infinities. The native short test reproduced the previous F32 logits, draft taps, cache hash and snapshot comparison. Four pinned 4K states matched 128 projected F32 logits, eight IDs and eight F32 weights at all 48 visits. The pinned prefill seed and first continuation matched in each case.

| Same-cache one-token boundary | Ordered scan median | Parallel selector median | Median paired saving | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Instrumented route selection and weights over the 44 NVFP4 visits | 2.611 ms | 0.626 ms | 1.973 ms | 8/8 |
| Plain whole command | 29.499 ms | 26.871 ms | 2.238 ms | 8/8 |

Both arms used the vector NVFP4 kernel. The first worker launch lacked the native template helper needed for startup image checks and yielded zero requests. The measured launch uses a locked Rust 1.98.1 build of the helper with the normal image checks enabled.

In the matched worker screen, four pairs per sampling mode used 4,096 input and 256 emitted and sampled tokens with zero prefix reuse. Both arms used the same binary, one-row projection, vector NVFP4, prefill policy and resident vision and DSpark. Every eligible output hash matched.

| Sampling | Vector NVFP4 alone | Vector NVFP4 with parallel selection | Median paired change | Selector range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 40.54 tok/s | **44.24 tok/s** | +8.69% | 43.65–44.52 | 4/4 |
| Seeded sampled | 38.28 tok/s | **41.08 tok/s** | +7.21% | 40.41–41.55 | 4/4 |

Paired prefill moved +2.39% greedy and +2.60% sampled. Worker GPU attribution was 99.86% over 95 intervals. The host recorded nominal and fair thermal states, with +0 pageouts and +0 swapouts. Keep the selector opt-in following this screen. One subsequent flags-unset pair per sampling mode matched the prior worker output hashes with zero reuse.

## 20261121-INKDECODE-05 · NVFP4 weight sharing across short verifier runs

Reuse one vector-decoded NVFP4 weight block for the active rows that route to the same expert. The short verifier uses one SIMD group per output channel and keeps an independent K16 FMA recurrence for every route, then applies the existing exact grouped gate/up and down epilogues. It avoids the BF16 ingress, padding arithmetic and replay in the bulk tensor path. Widths 2, 3, 4, 6 and 8 matched every F32 logit vector and the complete committed-state hash at the short prefix and both pinned 4K prefixes.

| Rows | Separate vector expert products, 4K | Grouped tensor expert products, 4K | Shared vector expert products, 4K |
| --- | ---: | ---: | ---: |
| 2 | 37.70–37.85 ms | 44.31–44.32 ms | 32.23–32.27 ms |
| 3 | 51.25–51.43 ms | 50.01–51.09 ms | 37.04–37.68 ms |
| 4 | 64.35–64.46 ms | 54.20–59.53 ms | **40.98–43.54 ms** |
| 6 | 93.65–94.00 ms | 67.61–76.11 ms | 54.67–58.39 ms |
| 8 | 121.42–122.11 ms | 80.33–89.85 ms | 65.96–70.41 ms |

Each endpoint is a median of four alternating same-prefix commands. Four-row profiles attributed 33.17–38.07 ms to the NVFP4 expert region with the tensor path and 19.97–22.42 ms with shared vector products. The 4K attention and relative cache regions remained about 5.9 and 5.5 ms. An initial run without grouped profiling reproduced exact parity and the same ordering across all widths.

At depth three, the earlier draft measurements allowed 16.79 and 30.36 ms for verification, rollback and host work on the two 4K inputs. The four-row commands still exceed those budgets. The shared-vector arm runs in the verifier diagnostic. Rejected-suffix rollback and worker integration remain open.

## 20261121-INKDECODE-06 · Screen Lily’s prompt lookup on Inkling continuations

Replay the same latest-occurrence 4-gram index and backward agreement rule used by Lily at each ordinary Inkling decode position. Require 32 matching tokens, or eight with a line break, and check up to seven proposed tokens against the continuation. The eight runs used the same 4,096 inputs, 256 outputs and sampling settings as the parallel-selector pairs. All eight output hashes matched. The replay kept the prompt and continuation transient and emitted aggregate counts and output hashes. [Lily’s lookup implementation](../../crates/stark-engine/src/engine/hybrid/mtp_lookup.rs)

| Sampling | Ordinary decode intervals | Positions with proposals | Positions accepting at least one | Inputs with zero proposals |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 1,020 | 15 | 13 | 2/4 |
| Seeded sampled | 1,020 | 7 | 6 | 3/4 |

The overlapping windows contain 49 greedy and 21 sampled accepted draft rows even when every proposed depth is considered. If every one removed a full serial step without any verification cost, the gain would be at most 5.05% greedy and 2.10% sampled across these runs. Overlap and the actual verifier cost reduce the gain. Defer lookup as a route to 60 tok/s on this workload. This replay leaves copy-heavy workloads unqualified.

## 20261121-INKDECODE-07–09 · Three single-token kernel screens

Each candidate used the same source singleton, vector NVFP4 and parallel expert selection as the control and changed one operation. The profile restored four stock 4K states, alternated the order and recorded eight instrumented and eight plain pairs per candidate. Short F32 logits, target taps and state hashes matched the preceding build. Each 4K seed, first continuation and all 48 visits’ router logits, IDs and weights matched the control.

| Candidate | Component median, control → candidate | Plain command median paired change | Faster plain pairs |
| --- | ---: | ---: | ---: |
| Fuse the independent QKV, relative, gate, sink and dynamic projections into one dispatch | 6.166 → 5.045 ms | 0.170 ms faster | 6/8 |
| Factor the E4M3 activation scale out of each exact NVFP4 block, then use four FP16 coefficient dots | 5.269 → 5.367 ms | 0.196 ms slower | 1/8 |
| Use a 32 × 16 × 16 TensorOps singleton router tile in eight groups, with the same K split | 2.344 → 6.175 ms | 4.183 ms slower | 0/8 |

Fusing dense projections removed 256 dispatches and saved 1.087 ms at the paired component median. The whole-command effect was under 1%. The FP16 expert trial stored FP4 coefficients and the shared scale separately. Four products of half-integers fit exactly in FP16 on the quarter-unit grid, and the scaled K16 accumulation stayed FP32. The short report’s raw quantizer comparison read the former F32 scratch layout, so those two quantizer taps need a decoder to compare the new layout. Every other reported short boundary matched. The narrower router tile reproduced the full projection output, but its longer component time precluded worker scoring. All three prototypes were retired after these screens.

## 20261121-INKDECODE-10 · Certified seeded draws with a sorted prefix

For full-vocabulary stochastic sampling with no effective top-p truncation, compute the existing F64 weights in exactly sorted order for a bounded prefix. The omitted logits must all be at least 16 log-weight units below the maximum under the original F32 temperature calculation. Their combined weight and rounding allowance bound the full draw threshold. Use the original seeded draw and accept a prefix token only if the entire possible threshold interval falls within that token’s original cumulative mass. Otherwise use the existing full sort with the RNG still at its original state. The flag is `STARK_INKLING_EXPERIMENT_CERTIFIED_SAMPLE=1`.

The sampler test verified exact token choice and RNG advancement across 127 seeds, including tied logits, signed zeros and flat or invalid fallback inputs. On one separately instrumented 4K sampled pair, 233 of 256 draws were certified and 23 used the full sort. Mean sampler time fell from 0.865 to 0.291 ms per draw. Greedy used the existing scan in both arms.

With the vector experts and parallel selector in both arms, the cold uninstrumented worker matched all eight output hashes. Every eligible window used 4,096 input and 256 output tokens with no prefix reuse.

| Sampling | Control | Candidate | Median paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Greedy control | 42.90 tok/s | 42.19 tok/s | +0.83% | 3/4 |
| Seeded sampled | 39.94 tok/s | **41.15 tok/s** | +4.28% | 3/4 |

Paired prefill moved +1.09% and +1.23%. The workers accounted for 96.22% of sampled GPU counter changes over 96 intervals. The host sampled nominal and fair thermal states with +469 pageouts and +0 swapouts. The sampler remains opt-in after the four-pair screen. Its measured CPU saving is about 0.57 ms and the 60 tok/s target needs further target computation savings.

## 20261121-INKDECODE-11 · Source-precision DSpark acceptance

Capture all four BF16 target taps for every packed prefill row. Feed each row into the existing draft cache commits, then run the stock 2,048 + 2,044 + 3 + 1 schedule with source precision. The first diagnostic attempt inherited the ordinary draft-mode chunk cap. The final-row tap check rejected that run before it recorded proposal timings. With the corrected chunk bound, both bulk commands matched the independent final-row tap capture on each stock fixture.

The four-token input matched the pinned first continuation. Both 4K inputs matched the pinned seed and first continuation, covering five F32 logit vectors total. All 96 first proposals matched the singleton draft, and the depth variants reproduced their shared prefixes. Teacher replay also matched in every draft round.

| Starting state | Accepted prefixes by proposal position, out of 32 | Median draft ms at depth 1 / 3 / 7 | Verify and finish budget at 60, depth 3 |
| --- | --- | --- | ---: |
| Four-token prefix | 26, 23, 18, 11, 6, 2, 0 | 4.45 / 5.15 / 6.60 | 46.41 ms |
| Stock 4K input A | 6, 0, 0, 0, 0, 0, 0 | 4.74 / 5.46 / 6.91 | 14.34 ms |
| Stock 4K input B | 26, 19, 12, 7, 2, 1, 0 | 4.80 / 5.53 / 6.97 | 40.83 ms |

Both continuations are greedy. These source-target results supersede the legacy-target draft acceptance estimate for planning an exact source verifier. Input B leaves a viable margin if the four-row verifier and finishing fits in 40.83 ms. Input A still needs a substantially cheaper ordinary token. The packed draft tap path runs in the native diagnostic. Useful worker decode and rejection rollback remain open.

## 20261121-INKDECODE-12 · Parallel selection in the short verifier

Apply the singleton’s exact SIMD selection independently on every verifier row. The shared TF32 router projection stays in place. The one-dispatch selector replaces the per-row scan and the separate log weight, softmax and scale dispatches. This arm uses the previous grouped vector expert products.

| Verifier rows | Same-state 4K command before | With parallel selector | Paired saving across the 4K inputs |
| --- | ---: | ---: | ---: |
| 2 | 32.38–32.41 ms | 30.27–30.32 ms | 2.11–2.12 ms |
| 3 | 37.04–37.76 ms | 34.89–35.60 ms | 2.15–2.18 ms |
| 4 | 41.09–43.62 ms | **38.95–41.52 ms** | 2.07–2.15 ms |
| 6 | 55.31–58.10 ms | 52.70–55.57 ms | 2.20–2.21 ms |
| 8 | 66.03–70.39 ms | 63.75–68.10 ms | 2.27–2.35 ms |

All four pairs per width on the short and two stock 4K fixtures were faster. Every checked F32 logit and committed-state hash matched at widths 2, 3, 4, 6 and 8. The final combined executable also reproduced the short singleton, legacy draft replay and stock B source-mode proposal acceptance. On stock B the four-row command sits about 0.70 ms above the depth-three verification and finish budget, before rollback and host work. The diagnostic remains ahead of worker integration.

## 20261121-INKDECODE-13 · Exact short-round rollback and worker integration

Save the local KV slots a verifier block may overwrite and the four convolution histories per visit. In a four-row block the last row can overwrite the accepted anchor’s convolution slot, so capture that anchor’s raw inputs during verification. Undo rejected slots while leaving the global future rows outside the committed length. The target writes draft taps for every verified row and commits only the consumed inputs to DSpark. Abandoning a transaction resets the live position.

The native controls covered every kept length at widths two, three and four on the short and both stock 4K prefixes. The two 4K fixtures repeated the matrix across the local ring wrap at position 4,095. All 45 cases matched the target and draft state and the next F32 continuation. Abandoned transactions reset the position, and restoring a source-draft snapshot in legacy draft mode failed. The first setup used a grouped expert workspace below its supported capacity. A subsequent proof attempted host access to private draft buffers. Neither attempt yielded timing. The final proof uses the required minimum workspace and a GPU blit for private draft readback.

| Native fixture, 96 greedy tokens | Serial with draft cache | One proposal | Two proposals | Three proposals |
| --- | ---: | ---: | ---: | ---: |
| Four-token prefix | 46.51 tok/s | 54.26 tok/s | 66.13 tok/s | 72.65 tok/s |
| Stock 4K A | **44.50 tok/s** | 43.43 tok/s | 40.42 tok/s | 35.39 tok/s |
| Stock 4K B | 44.35 tok/s | 45.13 tok/s | **47.99 tok/s** | 44.47 tok/s |

The measurement includes proposal, verification, acceptance and finishing. The serial control keeps DSpark cache writes active. Every schedule matched the serial token stream and final target and draft state. Two proposals on stock B averaged 5.25 ms for the draft, 36.52 ms for verification and 0.45 ms for finishing. The short result answers a separate prefix case. The cold 4K worker gates still decide whether the 60 tok/s target is reached.

The worker has an experimental source-precision verifier for text prompts of at least 512 tokens on M5 Max at contexts up to 4,352. Set `STARK_INKLING_EXPERIMENT_SOURCE_VERIFY=1` and select one to three proposals through `mtp_drafts`. The worker uses its ordinary sampler on each row as the consumer requests it. A matched proposal is delivered immediately, and replacement or bonus output commits the already consumed inputs. An early stop finalizes only the inputs consumed so far. Source precision and draft mode bind the prompt snapshot, and the worker retains no speculative position between requests.

The first worker launch reported the historical one-draft capability identifier alongside a maximum of three. Client validation rejected readiness before scoring. The corrected native worker and converter use a distinct short-verifier capability and preserve the existing single-draft contract. All eight request windows used 4,096 cold input tokens, context 4,352, zero prefix reuse, 256 sampled and emitted tokens, and resident vision and DSpark. One locked Rust 1.98.1 worker ran all four depths in balanced order. Vector NVFP4, parallel expert selection and certified sampling were available in every arm. All scored output hashes matched and agreed with the prior serial workload outputs.

| Sampling | Proposals per round | Same-loaded drafting off | Fixed-depth median | Median paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 1 | 43.77 tok/s | **44.09 tok/s** | +2.73% | 3/4 |
| Greedy | 2 | 43.77 tok/s | **50.22 tok/s** | +16.66% | 4/4 |
| Greedy | 3 | 43.77 tok/s | **51.50 tok/s** | +18.29% | 4/4 |
| Seeded sampled | 1 | 42.08 tok/s | **40.47 tok/s** | -3.02% | 1/4 |
| Seeded sampled | 2 | 42.08 tok/s | **42.22 tok/s** | +2.27% | 3/4 |
| Seeded sampled | 3 | 42.08 tok/s | **38.78 tok/s** | -4.95% | 1/4 |

The best individual scored 4K rates were 54.49 greedy and 43.99 sampled tok/s. Even choosing the best measured fixed depth separately for each input gives medians of 51.78 and 42.85 tok/s. The best greedy setting accepted 628 of 1,166 proposals. The best sampled setting accepted 497 of 1,042. Active draft cache work moved the paired prefill rate by -5.05% for greedy depth three and -6.23% for sampled depth two. The worker accounted for 99.84% of the measured GPU counter change over 183 intervals, with nominal and fair thermal states and zero additional pageouts and swapouts. The worker remains below the 60 tok/s target.

The final native build repeated all source stock B rollback and exact-state controls. Metal reported 57,278,464 additional allocated bytes for the verifier workspace and packed draft taps, within its 96 MiB admission reserve. The final worker stopped on a matched proposal before the round completed and matched the serial emitted prefix and sampled count. It reused a 2,048-token prompt prefix across proposal depths, compared exactly after changing the prompt tail, rejected reuse across draft mode changes, and reused zero tokens after cancellation during decode. Requests for three proposals on a 511-token prompt or at context 4,608 returned the scoped unsupported error. The final worker also matched one greedy and one sampled cold 4K pair against the prior worker with all Inkling flags unset.

## 20261121-INKDECODE-14 · Screen the next verifier and weight access changes

The existing packed local-attention routine in the short verifier changed a verified F32 row on stock 4K B. This screen used the full transactional guard, so it stopped at the first checked output difference and recorded no speed. The experimental wiring was retired. Any shared local-attention kernel for decode must preserve each singleton query’s shard assignment and the order of its FP32 updates, including rows whose old keys the block overwrites.

A separate read-only weight survey sampled 23,740,416 BF16 values from 168 physical attention tensors and 2,359,296 values from the vocabulary head. In 64-weight blocks, 99.35% of attention blocks and 99.43% of head blocks had a nonzero exponent span of at most 14 and no exceptional nonzero exponent. These sampled blocks fit one sign bit, seven fraction bits, a four-bit exponent code with one code reserved for zero, and one base exponent per block. That layout would read about 24% fewer weight bytes in the surveyed stream with exact BF16 fallback for wide blocks. This is an encodability result. Kernel cost and sustained speed are unmeasured.

## 20261121-INKDECODE-15 · Lossless BF16 attention weights on Metal

Pack blocks of 64 BF16 values into 64 sign-and-fraction bytes, 32 bytes of exponent codes and one base exponent. Reserve one code for signed zero and keep an ordinary BF16 block for exceptional values or a wider exponent span. Audit reconstruction on the CPU and reconstruct each BF16 value in an otherwise identical source matvec. The GPU kept the same K groups, FP32 accumulation order and BF16 output. Inputs came from a pinned stock 4K capture whose seed and first continuation matched.

| Same-loaded native control | BF16 words matched | Ordinary GPU | Packed GPU | Ordinary wall | Packed wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| One layer-11 attention gate, 20 pairs | 3,072 | 0.0616 ms | 0.0844 ms | 0.2107 ms | 0.2368 ms |
| 336 projections over 48 visits and 168 physical banks, 12 comparisons | 523,392 | 5.408 ms | 7.297 ms | 5.877 ms | 7.817 ms |

The rotating test estimated 24.1% fewer weight bytes, 2.582 GB to 1.960 GB including fallback loads. Paired GPU and wall costs rose 1.920 ms and 1.957 ms. Retire this reconstruction kernel. The result covers attention projections on one 4K input, with no complete worker change.

## 20261121-INKDECODE-16 · Share local KV reads in the exact short verifier

Rotate the shared absolute-key shard to each query’s original logical shard. Preserve each query’s chronological FP32 updates, softmax sink and output gate. Read overwritten old keys from the transaction’s undo buffer. With two queries per group, an unpaired row gets its own group. The initial all-row implementation matched the checked logits but ran one 4K depth-three native trajectory at 32.13 tok/s with roughly 60 ms spent in verification per round. The timing cause was unresolved.

In the second screen, eight comparisons at each width alternated candidate order on the same loaded model and snapshot. Timings begin before transaction encoding and end when verified logits are available. The measured range shows the two distinct 4K inputs. All of the checked F32 rows and 45 rejected-suffix and wrap controls matched across the short and two 4K fixtures.

| Verified rows | Baseline median | Two queries per group median | Median paired saving |
| --- | ---: | ---: | ---: |
| 2 | 30.59–30.62 ms | 30.28–30.36 ms | 0.27–0.28 ms |
| 3 | 35.28–35.95 ms | 34.86–35.50 ms | 0.43–0.44 ms |
| 4 | 39.35–42.00 ms | 38.90–41.35 ms | 0.52–0.60 ms |

A separate exact NVFP4 fusion shared ingress coefficient reads between grouped gate and up projections while keeping both BF16 roundings and the original nonlinearity. It increased paired verifier time by 0.61–0.69 ms at two rows and 1.07–1.19 ms at four. Combining the fusion with shared KV reads cost time over the shared KV arm. Retire the fusion. Retain two-query KV sharing via `STARK_INKLING_EXPERIMENT_SHARED_LOCAL=1` within the experimental source verifier. The final native build repeated all short and stock `g4096-01` controls. The verifier allocated 60,473,344 extra Metal bytes including draft taps, up 3,194,880 bytes from attempt 13 and within the existing 96 MiB reserve.

The final worker ran two cold 4K pairs per sampling mode with both arms on the same build, vision and DSpark resident, context 4,352 and zero input reuse. Each request sampled and emitted 256 tokens. All eight output hashes agreed with the ordinary target. Greedy used three proposals and sampled used two.

| Mode | Shared KV off | Shared KV on | Median paired decode change | Median paired prefill change |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 51.54 tok/s | 52.84 tok/s | +2.54% | +2.54% |
| Sampled | 41.06 tok/s | 43.73 tok/s | +6.52% | +13.00% |

All four pairs favored sharing on useful decode. Load order was reversed for the second input and prefill varied across the launches. Treat the worker rates as a small follow-up screen. The same-snapshot command comparison isolates the KV change.

## 20261121-INKDECODE-17 · Measure stochastic DSpark acceptance before changing decode

Collect 96 anchor positions on each of two stock 4K inputs using a source-target teacher sampled at temperature 1 with no probability filters. The pinned 4K seed logits matched bit-for-bit on both inputs. Export three DSpark score rows conditioned on the teacher continuation. Check argmax through all three masks against the normal proposal path at six positions per input. All 36 masks matched. Raw logits, tokens and draws stayed in diagnostic memory.

For an exact distribution-preserving rejection sampler, the one-position acceptance is the overlap mass `Σ min(p, q)` of target `p` and draft `q`. On target-sampled teacher tokens, accumulate the product of `min(1, q(token)/p(token))` through each mask to estimate joint acceptance. The sum of the three prefix probabilities plus one estimates useful output per round. Also compare selecting draft proposals with the ordinary target’s random number and vocabulary order, which preserves existing seeded output after target verification.

| 4K input | Draft temperature with highest estimated yield of four tried | Estimated rejection acceptance through masks 1 / 2 / 3 | Estimated useful output per round | Current greedy draft match counts on sampled teacher | Best common-draw counts at depth 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s4096-00` | 1.00 | 71.0% / 42.2% / 23.9% | 2.372 | 59 / 31 / 15 of 96 | 17 of 96 at 0.50 |
| `g4096-01` | 0.75 | 82.4% / 65.0% / 49.5% | 2.969 | 75 / 58 / 42 of 96 | 48 of 96 at 1.00 |

At 60 tok/s, the three-mask estimates leave 39.53 and 49.48 ms total per round. Subtract earlier draft medians of 5.53 and 5.46 ms and compare four-row verifier medians of 41.35 and 38.90 ms on the corresponding stock input. The first budget is already 7.35 ms short. The second leaves about 5.1 ms for correction sampling, rollback and host work. These estimates use 96 consecutive anchors per input and a single restored verifier state per input. They estimate the round-time budget for each input. Worker throughput is a separate measurement. Shared-draw counts vary with draft temperature and show little improvement on the harder input.

## 20261121-INKDECODE-18 · Table the exact source NVFP4 nonlinearity

The gate and up projections are rounded to BF16 before nonlinearities. Precompute the F32 gate term and capped up term for all 65,536 input bit patterns on Metal with the same precise functions. The candidate looks up the two terms and retains the source FP32 multiplication order, route weight and BF16 output. The table uses 512 KiB. The singleton vector dot and short grouped verifier used the candidate. Stock prefill continued to use the ordinary epilogue.

Eight alternating same-snapshot comparisons at each width matched all F32 logits on one short and two pinned 4K inputs. All 45 rejected-suffix and wrap controls matched. The two stock 4K inputs saved 0.067–0.160 ms per singleton command and 0.041–0.043 ms per four-row command at the paired median. Retire the shader and initialization. The full-command gain did not justify another runtime path.

## 20261121-INKDECODE-19 · Specialize the short NVFP4 product by expert run width

The grouped verifier previously used a four-route implementation for each run of two to four routes. Compile exact one-, two-, three- and four-route implementations and select by the live run size. Each route keeps the same original K16 dot, per-lane scaled FMA recurrence and final SIMD reduction. Runs beyond four retain four-route tiles plus the matching tail. Prefill and the ordinary singleton do not use this kernel.

In eight alternating same-loaded comparisons per width and input, both 4K fixtures and the short fixture matched every checked F32 readout and the 45 rejection and ring-wrap cases. The full 96-token greedy continuation and committed state matched at all three proposal depths on each fixture.

| Rows | Ordinary 4K median | Exact run width 4K median | Median paired saving |
| --- | ---: | ---: | ---: |
| 2 | 30.49–30.50 ms | 29.80–29.96 ms | 0.54–0.65 ms |
| 3 | 35.26–35.85 ms | 34.58–35.21 ms | 0.65–0.67 ms |
| 4 | 39.29–42.06 ms | 38.66–41.08 ms | 0.69–0.91 ms |

Retain the specialized kernel within the source verifier under `STARK_INKLING_EXPERIMENT_EXACT_RUN_WIDTH=1`. The native comparison isolated the kernel with shared KV disabled. The worker follow-up uses shared KV on both arms.

Two worker pairs per mode used the same sealed executable, with shared local KV on, 4,096 input tokens, 256 sampled and emitted tokens, zero reuse and both vision and DSpark resident. Greedy used three proposals and sampled used two. All eight output hashes and proposal and acceptance counts agreed within pairs.

| Mode | Ordinary run kernel | Specialized run kernel | Median paired decode change | Median paired prefill change |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 49.31 tok/s | 45.95 tok/s | -6.63% | -6.23% |
| Sampled | 39.26 tok/s | 37.17 tok/s | -5.19% | -10.29% |

The specialized arm was slower in both worker pairs per mode. The first input also had substantially slower prefill in the specialized arm, even though the specialization operates only in verification. The second input had near-flat decode. Treat this as an adverse worker screen with concurrent launch drift. The same-loaded native test qualifies only the measured command boundary.

A follow-up native comparison enabled shared KV in both arms. The run-width specialization still saved 0.71–0.73 ms at four verifier rows across the two 4K states. With both paths enabled the 4K medians were 38.11 and 40.38 ms. Pinned readouts and rollback matched again. Keep that result at the command boundary given the worker screen.

With all Inkling flags unset, the final native executable reproduced the connected short and 4K slice aggregates. The full target matched the pinned short and 16 stock 4K F32 logit vectors, 768 stock visit residual blocks and four exact 4K snapshots. One 4K flags-unset worker pair per mode reproduced the promoted router build's output. The candidate measured 33.39 tok/s greedy and 32.05 tok/s seeded sampled, against 33.45 and 32.33 tok/s on the promoted reference. The eight-pair [default router qualification](2026-11-21-inkling-mini-decode-default.md#worker-results) remains the stronger throughput measurement.

## 20261121-INKDECODE-20 · Replay Lily’s depth planner with Inkling round costs

Use the existing request-scoped depth planner with stock Inkling M5 Max command costs and rounded draft, finish and host allowances. Keep the exact ordinary target draw and feed the planner only completed round depth and accepted count. A zero-depth round still commits its target token and updates the source draft cache. The trial opened with a three-proposal cap and used the existing rollback and output paths.

Four alternating cold 4K pairs per mode used the same sealed binary, with shared local KV and exact run width on in both arms. The fixed arm used three greedy or two sampled proposals. Vision and DSpark were resident. Every request sampled and emitted 256 tokens with zero reuse. All 16 output hashes matched the ordinary target.

| Mode | Fixed useful decode | Adaptive useful decode | Median paired decode change | Faster adaptive pairs | Median paired prefill change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 54.42 tok/s | 53.30 tok/s | -1.99% | 0/4 | -0.44% |
| Sampled | 44.12 tok/s | 43.47 tok/s | -1.73% | 1/4 | -0.41% |

The planner issued [1, 3, 73, 326] greedy and [165, 7, 288, 132] sampled rounds at depths zero through three. GPU client counters attributed 99.83% of observed increments to the two workers over 104 intervals. Thermal samples were nominal and fair. Pageouts changed by +155 and swapouts by +0 pages. Retire this planner integration after the screen. The fixed paths remain available within the experimental verifier.

## 20261121-INKDECODE-21 · Store exact verifier NVFP4 ingress in binary16

The activation quantizer emits FP4 half-integers multiplied by finite positive E4M3 scales. Each exact product has at most eight significant bits and a nonzero magnitude between 2⁻¹⁰ and 2,688. Store those coefficients in binary16 and widen to F32 at the grouped verifier dot. The dot grouping, scaled FMA recurrence, SIMD reduction, route-weight epilogue and BF16 rounding stay in source order. The scratch allocation still fits both formats.

Alternate seven arms on the same restored snapshot at widths two, three and four. The eight comparisons per arm and width checked every F32 row on both stock 4K inputs and the short input. The first Metal compilation failed because the activation vector pointer needed an explicit device address space. It yielded zero timed rows.

| Verified rows | F32 ingress median, 4K | Binary16 ingress median, 4K | Median paired saving |
| --- | ---: | ---: | ---: |
| 2 | 29.60–30.47 ms | 29.32–30.23 ms | 0.23–0.47 ms |
| 3 | 34.70–34.79 ms | 34.22–34.37 ms | 0.34–0.56 ms |
| 4 | 38.78–40.48 ms | 38.13–40.00 ms | 0.47–0.67 ms |

One, two and eight SIMD groups per threadgroup with ordinary ingress had mixed timings. Eight SIMD groups with binary16 ingress was slower on one stock input and near even on the other. Polling Metal completion on the four-SIMD F32 arm saved about 0.02–0.06 ms at the paired median on 4K. Retire those variants. Keep the four-SIMD binary16 candidate under `STARK_INKLING_EXPERIMENT_VERIFY_HALF_INGRESS=1`. The subsequent shortlist build also repeated the short and two stock 4K rejection, ring-wrap and committed-state controls with binary16 ingress on.

## 20261121-INKDECODE-22 · Rerank a smaller set of DSpark vocabulary candidates

Each SIMD group chooses the two highest base BF16 logits in 32 consecutive vocabulary entries, with ascending token ID to break ties. Only those candidates run the existing BF16 Markov correction and final softcap. This changes proposal selection. The target still samples ordinarily from exact logits and verifies every proposed row. Compare the full shortlist and contiguous vocabulary caps of 65,536 and 32,768 against ordinary full reranking at 64 consecutive greedy anchors per input, with arm order rotated within each anchor.

| Three-proposal arm | Median paired saving, stock 4K | Accepted prefixes through masks 1 / 2 / 3 on `s4096-00` | On `g4096-01` |
| --- | ---: | ---: | ---: |
| Ordinary full rerank | — | 45 / 31 / 19 | 37 / 8 / 0 |
| Full-vocabulary shortlist | 0.80–0.81 ms | 45 / 29 / 17 | 37 / 8 / 0 |
| First 65,536 vocabulary entries | 2.20–2.23 ms | 10 / 0 / 0 | 0 / 0 / 0 |
| First 32,768 vocabulary entries | 2.56–2.59 ms | 10 / 0 / 0 | 0 / 0 / 0 |

The full shortlist changed 6–8 of 64 three-proposal rounds on the stock inputs. It saved 0.24–0.25 ms at depth one and 0.53 ms at depth two. The native continuation and complete committed state matched the serial target on the short and both 4K cases. Drop both fixed vocabulary caps because they removed most first-mask acceptances.

One sealed build compared the full shortlist plus binary16 verifier ingress against the previous full rerank plus F32 ingress. Both arms used exact run widths and shared local KV. Vision and DSpark were resident. Four alternating cold 4K pairs per mode used context 4,352, zero reuse and exactly 256 sampled and emitted tokens. Greedy used three proposals and sampled used two. All 16 output hashes matched the ordinary target.

| Mode | Control useful decode | Combined useful decode | Median paired decode change | Faster combined pairs | Median paired prefill change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 52.70 tok/s | 53.05 tok/s | +2.48% | 4/4 | -0.74% |
| Sampled | 43.12 tok/s | 43.32 tok/s | +1.97% | 3/4 | +0.35% |

The best request reached 57.41 tok/s greedy and the best sampled request reached 44.38 tok/s. Greedy accepted 623 of 1,181 proposals with the candidate versus 628 of 1,166 in the control. Sampled accepted 489 of 1,058 versus 497 of 1,042. Each stock worker output contained 156–189 tokens with IDs at or above 65,536. The workers accounted for 99.79% of GPU client counter increments over 106 intervals. Thermal samples were nominal and fair, with +81 pageouts and zero additional swapouts. Execution rates varied by request and session. Retain the full shortlist as `STARK_INKLING_EXPERIMENT_DRAFT_SHORTLIST=1` and keep both binary16 ingress and the shortlist opt-in. The worker comparison measures their combined effect.

The probability-overlap diagnostic always uses the ordinary full rerank. With the shortlist worker flag set, its stock 4K proof reproduced the previous aggregate receipt byte-for-byte, including 18 draft argmax controls.

## 20261121-INKDECODE-23 · Share NVFP4 ingress across adjacent output columns

Each SIMD group loaded one binary16 activation block and applied it to two adjacent output columns, with independent F32 recurrences and source reductions. Test the up projection alone, the down projection alone and both in eight rotated comparisons per width and input. The initial Metal build needed explicit vector widening and produced no measurements. Every checked row matched the short and both pinned 4K controls after that repair.

At four verifier rows, pairing only the up projections added 0.76–1.00 ms at paired medians on the two 4K inputs. Pairing only the down projections added 0.51–0.68 ms. Pairing both added 1.22–1.58 ms. Retire the two-column shader. This measures the complete verifier command with binary16 ingress in all arms.

## 20261121-INKDECODE-24 · Preserve source accumulation around TensorOps K16 dots

Use cooperative FP16 input fragments and an F32 destination to regroup each exact 16-value dot. Decode packed FP4 weights into the fragment registers. Widen binary16 ingress through the matrix instruction. Keep one partial per original source lane and output, applying the E4M3 weight scale with the original FMA order over blocks lane, lane + 32 and so on. Finish with the same SIMD reduction. The Metal SDK requires 16 or 32 matrix rows when both input operands are cooperative tensors. A four-row physical tile failed compilation and yielded no timings. The tested 16-row tile pads the two to four live verifier rows.

With four alternating comparisons per arm at verifier widths two and four, all checked F32 row vectors matched on one short and two stock 4K inputs.

| Verified rows | Vector control, 4K | K16 TensorOps with four SIMD groups | K16 TensorOps with eight SIMD groups |
| --- | ---: | ---: | ---: |
| 2 | 29.11–29.31 ms | 83.80–84.05 ms | 82.20–82.52 ms |
| 4 | 37.31–39.73 ms | 112.10–124.54 ms | 107.09–121.64 ms |

Retire both TensorOps configurations. At this geometry, one matrix operation per K16 block and preserving 32 separate source-lane recurrences was much slower than the vector control. This result covers the entire verifier command, including the unchanged binary16 ingress and expert epilogues.

The retained build reproduced the flags-unset connected short and 4K slice aggregates, the pinned full-target short logits, and 16 pinned 4K F32 vectors across four inputs. All 768 visit residual comparisons and four 4K snapshot replays passed. With both new options enabled, the native build repeated all 45 rejection and cache-wrap controls and the exact serial continuation and committed state on the short and two stock 4K inputs.

## 20261121-INKDECODE-25 · Replay short-context lookup with past continuation agreement

Replay the four greedy and four seeded sampled stock 4K outputs using only earlier tokens to select proposals. The sealed ordinary worker produced 256 sampled and emitted tokens per request with zero reuse, and all eight output hashes matched the prior ordinary controls. For each current suffix of one to four tokens, test agreement among earlier continuations. Require support from two or four past occurrences and consensus of at least 75%, 90% or 100%. Test caps of one or three proposals. The replay advances by the accepted prefix plus one output, so overlapping windows cannot each claim the same saved work.

Charge 22.8 ms for a serial step and 29.5, 34.6 and 39.2 ms for verifying one, two or three proposals. These rounded costs come from the measured vector and binary16 configurations. Charge zero time to build and query the lookup table for this screen.

| Mode | Best median rule in the screen | Lookup rounds | Accepted proposals | Estimated gain over serial |
| --- | --- | ---: | ---: | ---: |
| Greedy | One-token suffix, unanimous support from at least two occurrences, up to three proposals | 76 | 55 | +2.23% |
| Sampled | One-token suffix, unanimous support from at least two occurrences, one proposal | 71 | 31 | +1.20% |

The best greedy rule went from -0.84% to +4.55% across the four inputs. The best sampled rule ranged from -0.71% to +2.39%. The replay considers lookup in place of an ordinary token. The existing DSpark worker already runs faster than that ordinary reference. Defer runtime integration based on the small estimated gains and sparse eligibility.

## 20261121-INKDECODE-26 · Compact live NVFP4 runs in the short verifier

Sort at most 32 routes in one SIMD group by expert and original route index. Generate the same stable run records and a Metal indirect grid with one column of workgroups per live run. The control scans all 128 experts and launches one column per route. Compare the original, SIMD sort with the static grid, and SIMD sort with the indirect grid on the same restored state.

| Verified rows | Static control, 4K median | SIMD sort and indirect dispatch, 4K median | Median paired saving |
| --- | ---: | ---: | ---: |
| 2 | 29.22–29.47 ms | 29.09–29.27 ms | 0.13–0.29 ms |
| 3 | 33.73–34.19 ms | 33.47–33.97 ms | 0.23–0.31 ms |
| 4 | 37.46–39.84 ms | 37.20–39.55 ms | 0.28–0.29 ms |

Eight rotated pairs at each width reproduced the F32 logits on short and both stock 4K captures. All 45 rejection and cache-wrap controls and all 96-token serial continuation and committed-state checks passed with indirect dispatch enabled. The paired saving from the sort alone was 0.11–0.13 ms at four rows. Enable both pieces under `STARK_INKLING_EXPERIMENT_VERIFY_COMPACT_RUNS=1` in the short source verifier.

## 20261121-INKDECODE-27 · Batch convolution and residual finishing across verifier rows

One thread per channel walks provisional rows in causal order. The expert kernel reduces the eight top routes, rounds the merged value to BF16, writes the original three-slot convolution history and updates the residual at the existing BF16 boundary. It also writes the merge needed by the anchor rollback. The attention kernel batches the attention projection’s source convolution and residual update, preserving the old partial when a cross-pass boundary opens. Each kernel preserves the scalar source multiply/add order. The trial uses the reference dispatches for instrumented layer captures.

| Kernel enabled | Four-row control, 4K median | Four-row candidate, 4K median | Median paired saving |
| --- | ---: | ---: | ---: |
| Expert finish, with compact runs in both arms | 37.22–39.74 ms | 36.67–39.20 ms | 0.55–0.56 ms |
| Attention finish, with compact runs and expert finish in both arms | 36.66–39.09 ms | 36.13–38.54 ms | 0.54–0.56 ms |

Eight alternating pairs per width matched every F32 row on all three native fixtures. The complete short and 4K continuations stayed exact with both fusions on, including 45 rejection and cache-wrap cases and exact committed states. The effects are additive at the measured command boundary. The experimental controls are `STARK_INKLING_EXPERIMENT_VERIFY_FUSED_FINISH=1` and `STARK_INKLING_EXPERIMENT_VERIFY_FUSED_ATTENTION=1`.

Four alternating pairs per mode compared the prior worker configuration against compact runs and both finishing fusions. The two arms used one build, the same 4,096 cold input tokens and 4,352 context, zero reused tokens and 256 sampled and emitted tokens. Both arms used the prior half-ingress and draft-shortlist controls. Vision and DSpark were resident. All 16 output hashes and paired acceptance counts matched.

| Mode | Control useful decode | Combined useful decode | Median paired decode change | Faster pairs | Median paired prefill change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 54.47 tok/s | 56.32 tok/s | +2.58% | 3/4 | +1.29% |
| Sampled | 43.87 tok/s | 43.12 tok/s | -1.45% | 1/4 | -0.59% |

The combined arm peaked at 57.18 tok/s greedy and 46.37 tok/s sampled in this window. The sampled worker results varied across pairs and the median was slower. Keep the changes experimental based on the exact same-state command savings and the greedy worker gain. The workers accounted for 99.83% of GPU client counter increments over 103 intervals. Sampled thermal states were nominal and fair. Pageouts changed by +23 and swapouts by +0 pages.

## 20261121-INKDECODE-28 · Screen held-out seeded calibration and sparse rejection

Generate 192 sampled anchors on each of four different pinned stock 4K prompts at temperature 1. Each trace starts with the pinned F32 seed logits. Compute proposals conditioned on the matching teacher prefix. Replay sequential, nonoverlapping rounds using measured favorable costs of 22.8, 34.4, 39.7 and 44.0 ms at proposal depths zero through three. The costs leave out additional calibration, probability and rejection implementation overhead.

For the exact seeded stream, choose between the top two draft tokens using a shifted interval of the ordinary target’s uniform draw. Learn the boundaries by mask position and eight bins of draft top-token confidence on the other three prompts. Fit with at least 48 training observations and a three-hit improvement over always proposing the draft argmax. The held-out draws and tokens do not enter the fit.

| Exact-seeded proposal choice | Best fixed-depth median, estimated tok/s |
| --- | ---: |
| Draft argmax | 51.02 |
| Shared uniform and full draft CDF | 50.45 |
| Held-out rank calibration | 50.69 |
| Top-two oracle replay | 59.60 |

The oracle chooses the correct target token whenever it is one of the draft’s two highest ranks. The calibration failed to improve the real choice. Retire the calibration experiment.

For a separate policy that preserves the output distribution, calculate each proposal’s conditional acceptance on teacher token `y` as `min(1, q(y)/p(y))`. The replay branches on that probability and charges the same fixed costs. Compare top-4, top-8, top-16, top-32, top-128 and full-vocabulary draft supports and three draft temperatures. The strongest fixed choice used depth three, temperature 0.75 and top-16. Its estimated median was 56.46 tok/s, versus 56.44 with full support. Both fall short of 60 before extra work to draw, verify and correct proposals. The result supports keeping the probability vector small if the path is revisited. A distribution-preserving worker would have a separate seeded sequence contract. Defer changing the worker sampler.

## 20261121-INKDECODE-29 · Shorten the exact FP4 activation threshold search

The FP4 quantizer selects from eight magnitudes using seven ordered midpoints. Two binary-search comparisons cross odd-indexed midpoints and round upward on a tie. The last crosses an even-indexed midpoint and rounds downward on a tie. Keep the sign and the original F32 accumulation of quantization error. A scalar check compared 66 signed values covering ties, the neighboring F32 values, zero, subnormals, infinities and NaNs.

Eight same-state comparisons at each width reproduced every F32 readout on short and both stock inputs for the scalar and vector candidates.

| Rows | Previous half-ingress control, 4K | Scalar threshold tree, 4K | Median paired saving |
| --- | ---: | ---: | ---: |
| 2 | 28.61–28.73 ms | 28.44–28.60 ms | 0.14–0.18 ms |
| 3 | 32.63–33.18 ms | 32.50–33.07 ms | 0.05–0.11 ms |
| 4 | 36.10–38.57 ms | 35.93–38.31 ms | 0.17–0.20 ms |

Processing four coefficients together slowed the four-row verifier by 0.05–0.09 ms. Retire that vector trial. The scalar tree now runs within the existing experimental binary16 ingress, so the ordinary quantizer keeps its previous behavior. The worker measurements in attempt 27 precede this scalar change.

The final retained executable passed the flags-unset short and 4K slice comparisons, the pinned full-target short logits, 16 F32 vectors and 768 BF16 visit residuals across four stock 4K inputs, and four exact snapshot replays. With the compact dispatch, both fusions and binary16 ingress all enabled, the final executable passed all 45 rejection and cache-wrap controls and matched the 96-token serial output and committed-state checks for short and both stock fixtures.

## Decision and next target

Keep the vector singleton, transactional verifier, shared KV, binary16 ingress and shortlist opt-in. The measured acceptance and round costs leave the harder sampled input short of the 60 tok/s budget. Prioritize reducing ordinary target and verifier execution time across inputs before changing the production sampler. Distribution-preserving rejection would need separate qualification because it can change the sequence for the same seed. Keep the existing seeded path available for exact replay.

## Evidence identities

Original target SHA-256 `65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1`. Original DSpark SHA-256 `7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2`. The current run uses the pinned Ollama capture and host documented in the [default-router record](2026-11-21-inkling-mini-decode-default.md).

| Artifact or private receipt | SHA-256 |
| --- | --- |
| Verifier-only executable | `91da72b4c4dae1dd4b221ddfcf67561964d11b923404007d3a17d1890775a244` |
| Multi-mask draft executable | `c182deda5762accd87b850a84b3f3cd1ac38516c923f3685f8aafa40ecd5cbd7` |
| Grouped and vector screen executable | `248d1eb2027b26150a1036967b490ee5c247292ea6483e84a2eedb5b6f749e5b` |
| Sealed measured worker | `537fe7ed1a2c0628d66db7a5de3025736e456ee4ea7be9f9c88260ab06d29a20` |
| Scored source patch against a370a786 | `e79893653a4d0be3377195b274ed5eab764ff0909e1d9340027770f814d6468b` |
| Stock selector profile | `2562a872739590c9073cb40e452df60abb9716df0ca51b91c8cd85cf204370e7` |
| Paired worker harness | `532fb44b9a52516e2a3fee89f0e2ff797189b60b8642ad7807651c1dab14e827` |
| Worker pairs | `ca06fef37167e8451e140f7f8a67c9cdb5a12824bb031d32ff68da93ccd6844b` |
| Host observations | `97c7b6ceb2665e56e02ea22d00f3705bf4e85a25d10f8fbfaae818513be10950` |
| Prefix lifecycle | `65ea98c3edb152d303e3a675be21f521ce9615a806846de493998201df74e4fd` |
| Packed tails | `f431d8684afff1117d665c127767293d734d8ef35db4b5c4763f908ae5b81da5` |
| Screen short | `8bf1eab0d63aabb79c5cc29af0c5f3fd2a7276c611eac6268c90f751afe06fe0` |
| Screen stock-schedule-g4096-01-retry | `edd5a3e5bb08b9c1f905ad0c6aba98a29166be61ba9e8c0bf0e5d17edb94e9a4` |
| Screen stock-schedule-s4096-00 | `32f738f77176c0f5d026f7f8fb92fd7086fdaadb0ea2dc3b1af896bc14f49661` |
| Draft short | `52a628c7b9de8ac93a3e916c51b743f83b915a69d1277a1b689e078d57cc99aa` |
| Draft stock-schedule-g4096-01-retry | `072d014911b027ca1f99d46453dbad47bc00435be7933fab023f9de2a9eda384` |
| Draft stock-schedule-s4096-00 | `3fab844e9eba837cdb6df476b59b2c5b19704e1ed1bd4683280cb95426a3d95a` |
| Final worker | `15514ff0ee8a14e9665fef990fafeb4d8bc751d52b2fbbe40bd23e37c67da3c3` |
| Final diagnostic | `3f95a28935912a49151dff90e02856454e6282897c88df95844c4565d628bcd2` |
| Final draft short | `5c065c27b688c68a2786def87a1217bc3f1beab5bc19aae22a79ace5ababcca3` |
| Final vector short | `d1a6bd0b8aa2faa77d7853dbb730c348f6e0c98042a425a6bed6f4e301f268b7` |
| Final vector 4K | `fd2423f86fef11ecc5e61715f38311aba33e67279b03cca2dc9ec36fea3e7712` |
| Final worker smoke pairs | `d869bc8eceb4354df54f09f70e1e2a7cdfe9d63925addded2a8561ee5aab3e4a` |
| Selector source patch against ab4d108f | `053858bf568126c84dccdd72ce152c26ebcac939cdc6f3a5715264aef002b072` |
| Selector native example | `401f2407aa4dc466acc068b75de294b326f1cedf776d1913758ee6bcd63702a2` |
| Selector sealed worker | `539c2d77eee51fe5a4e8f3309403fecd76c24473912fa5f295f7c8b2a1ed1d9a` |
| Selector same-cache profile | `cab0af2c541636be0ec9b63acd0af382236dbf37f3fe542c963023a15da169af` |
| Selector short parity | `05bb62f4e3a366aeee266be4a2c19185d4a0237767d57be4164fd30787b20f08` |
| Selector synthetic oracle harness | `c4f3e781896cc189bdf2326cbd6e907037daadb0e96d8cde39de03a8784bce19` |
| Selector synthetic outcome | `76890bf975f191e6aa306fc096e278bb4e625228b0d0b9d8e10ee927e312b3dd` |
| Selector worker pairs | `575bb91241f883a118a22c78653b010d81e9db1c085d1e3963da5453e82b34ae` |
| Selector worker host observations | `890adc80eac76d9989d23a32eca76ef9df2acab1fdb1ce23cdb55de8c957d154` |
| Selector worker harness | `285de90e6ae2bb9a3178407a48372692a893c528498166a90015a58303f8c5dd` |
| Selector flags-unset regression pairs | `5d9c1f7a3025b94479157fdc86cf180d079b7f439a094894e46f6c63cdc2c34a` |
| Short-run verifier source patch against 8604c5ae | `86c4dc3916ddd0670218bff091bab6a628a1ab01e4f9636815ad241c63afa4a9` |
| Short-run verifier executable | `6d84eae9d4b1a2977486f76b5b6bfd6c3ff71313bda8ff9d7dd7d43f09a3999e` |
| Short-run verifier four-token proof | `8c069d54f5915d4c1fb8afe2b9d75b1ff636c3c4ac6064f7bbe2346292b789dd` |
| Short-run verifier stock A | `0a7278d79d2827de8edd68be3fcb4eda00217b74cab8a2c97943ee9eae3c0da6` |
| Short-run verifier stock B | `ca530205f959d06ebdf644572ac20cc35652687b6006360b6090d6ac69c5d602` |
| Prompt lookup aggregate receipt | `ff7f185d416620bbe62ec6d6c5cb9359376419f1b33f2b69465f320553533509` |
| Prompt lookup replay harness | `58f5766d3ca99111edcf47a3a168757417103061824b1b9b6af656b901bf2b45` |
| Fused QKV source patch | `01e37f3f48af1741f83fe8bc48940202b1652bcaa0c4bc3a35ca2ccda1d7eb66` |
| Fused QKV executable | `cff28e72bf081d39697f41cdcdebbfb8d2cc97806bccedd9906b65504037adb7` |
| Fused QKV short receipt | `62d6d350cf6e7b40927f327ec738211bfce8b9126138c7def1d6b7a736d45382` |
| Fused QKV 4K profile | `a55d6368c119ffded5d731169f79d575e0510c0bff4735002eb1a110512090ae` |
| Half NVFP4 source patch | `7b4ef4e783816940aa084e4e7bbb0f32e9430902c878f89f4686d58b0af1168d` |
| Half NVFP4 executable | `2667b17ed3f69e398a7a0d571084b20f810c378f5515a8832a2db03338616ee6` |
| Half NVFP4 short receipt | `1cde5e5e5cda6653e4acec917fb3f3304373bff92b8c3b0e65d370b01fb99b7f` |
| Half NVFP4 4K profile | `8336cfdaadc7b7924b1ef5f6669305d080296b8f4c86b2110a43bf6432ff9e39` |
| Thin router source patch | `f141c3786a65a7c1236e5f6c1d150bfd7042a278843de2ca0407ceb9ea770518` |
| Thin router executable | `fbdabff8ac4936426208ededee63a372cd00fb39779c728c56d5ed18bb61e43b` |
| Thin router short receipt | `73a0c56e36ff833535d60cbf1413d191efa2a9614c71dd5ba0563def0537e458` |
| Thin router 4K profile | `4ac89a04634280100620b73bf8a9e6ddde49aee548c36276c496c8bea1737522` |
| Sampler source patch | `7b304d77efe070d3b9af42082e1fcd28ce54d128563ff81addc5e1898aa66bcc` |
| Sampler worker | `72a3e72f2cedce39f7fb3ee7bca2c0e2d0fa8c30fede8462fe48931a11b85b58` |
| Sampler worker pairs | `709bf93cf8ae213b9e5b0272d9fe82e6ca360d4675fd49c1eba3da50575a79e3` |
| Sampler worker host | `7d23a1b94938d11fb7eb8b04de4b42b7be92218f6d14fe65c10947d9189c2b5f` |
| Sampler worker harness | `8b7e7f4fae5eb0db29276dbc41edc9974f74b90c35a44caf8df96a3608f26b65` |
| Sampler profiling patch on candidate | `48d6bd9548e151fcf8fc548dd198eb01d937b5d51cf773eaf9d03ce144d147e6` |
| Sampler profiling worker | `1476805738e302ed02227dacff7a631671338eaca5978c557694a97d0a216811` |
| Sampler baseline scopes | `54dc22d9efe28e4aaca32e2018345b6c9b4be19524fe94503f740a04ce8d4501` |
| Sampler candidate scopes | `42e7da131c20a8116ad29eb38e81f2582407c226cb7ed203b8633d93d6b7210b` |
| Sampler instrumented parity pairs | `8638bfb70345e34016e9f2ac7dc5bf5150d2d4bb0a639efebe9d0691fbaf9d98` |
| Source DSpark patch against 7566c8ec | `d4db0239de94f208917e48c0d100441deef3d767f3fc3e72bb67970e04b160da` |
| Source DSpark executable | `3530ca55321eee177b41ad1bb183ee612656a2d31b27be3f184260d552f15059` |
| Source DSpark short | `64f184a4b7315754708f54264da5fe7fd4509561611150b22d1a0e25c5fcf7db` |
| Source DSpark stock A | `1074399b8a11f4ba2d4ff07bb38cc08bd8e27944c82e73ba94dcdf28ea72dd70` |
| Source DSpark stock B | `4a0e021c196f0527f3c469c13c55872d82dcfad1fc05b27d4e0194b352faa2fa` |
| Source DSpark invalid chunk cap | `c75ac078cf4f21383581562abe8a9981d402fbf63ed786dd699178135e4b16fc` |
| Parallel verifier patch against 7566c8ec | `533697c1642e2781894560dab1d1f27219b7bb36582c597b3bf788af8b5558df` |
| Parallel verifier executable | `1dd3da850f6b013a45e4a8ef6ae1e47c516365873b375874ae43c747a662d3c4` |
| Parallel verifier short | `658a92960b6c8ec3ecda2dd1163856aa817334a3f2c9945df37a2cadaa4f5f48` |
| Parallel verifier stock A | `fc5968b83e349fe99927b969869d15fda17450fdf418df1e3adf467c262dd8b5` |
| Parallel verifier stock B | `cca52b8917bf530a9e2f15225deb2ad0d1adbc113804c224711d6d9a2e2e78a2` |
| Final source draft stock B smoke | `97ff06f20db23615a834a3b14864dc315e522c559e71338a1e5264a4de9857b3` |
| Final legacy draft short smoke | `04e53b4bb380131bf1e18634ea91ed5c7ed2d46f1d2ca06aeeb7c2b8474ee8e5` |
| Final short singleton smoke | `e802ef9a3f66b9e9e80bd93f0ed08a54a75c9b85e6c1f29824073621b9e6a586` |
| Transactional verifier native patch | `2c64fe0c7b654f434f4ae65eb2de8fd1f07349368c5360f4311d689e55770200` |
| Transactional verifier native executable | `8adb1666d2646a5a216d3b9c6ecddbdd17c87f693990b9898674ee1a11b82a41` |
| Transactional verifier short | `14f510aec48124a914be11ff4299d0da707b1542d4409a484bc9264119dedadb` |
| Transactional verifier stock A | `448ec0cae07af65c893252e4dfeecd71020b025f5f810e31dadc1b0fcffb6a50` |
| Transactional verifier stock B | `5426040b94d00e7ebb45bed8b54fe2d301ca7795313abcf28f191b94aa3c6326` |
| Transactional verifier invalid grouped capacity | `f98471a41fe097a17b7f163718c84ef9e072d50d04d636730d902a071260112d` |
| Transactional verifier worker source patch | `10eace33e3cd44006dd3ecb8ec6eb058fdae2155484cc33f98062cb169b0c2e3` |
| Transactional verifier scored worker | `1378e36d906c0152a2c9156d0601d61cefc7b4690f89528be928db51756f708f` |
| Transactional verifier worker harness | `4caaa236b4bb1d2ad712d2f3c19f08d53588ffe7c3c738c9b9b2dfb9c62c1001` |
| Transactional verifier worker observations | `905dd2054d74fdc6e783f27313ca7c17ff50c2f733c72310f5c45dcfdb54d180` |
| Transactional verifier host observations | `eb874cf4076dc797247b95df6fd529934dd264e23621dc43cc1a0d701680545e` |
| Transactional verifier invalid client capability | `b754b166fe5a3e574979bf67197e3de4cc8be5b86313eed1bde26bf4b2a2a727` |
| Transactional verifier final native executable | `73152643776887218f65e1620a6b0a403e29a954bf6e15013efce5eb9f894557` |
| Transactional verifier final native recheck | `e4782ea0af8153a1661dce305a350420b239a93fa00ba2646143a647e55c9f1a` |
| Transactional verifier final worker | `a4fc10114ce2b0c93dff596a95d3cf26bd9914a779c0211c9e8743fc1b4927f1` |
| Transactional verifier lifecycle harness | `c41ce21f76b7170838513f7d1ee4bbaf09979d458e5ccf04249958e3fa18a06f` |
| Transactional verifier lifecycle controls | `d1b51cd09652bd043eb12d6546e89535af29ed286b5f587400a88200975afc6b` |
| Transactional verifier final source patch | `4fef4a81c7500da8aa9b5f10e18d57a195b241ceee488532dbe5e8181ba5b809` |
| Transactional verifier flags-unset worker pairs | `2ccfe3df802b05f4fdc9a4bdf6ea53d538c009136ffdad4474d08db59f33dc6f` |
| Packed relative trial patch | `4faf9d36d09f3a57b0ec61f85b4faa9e69f8f0a5b55e0cbacb881961521da0a5` |
| Packed relative trial executable | `0bff1a2c99dad62ce4b2dd1df8caf24f244abaf590f7e1b9d0eac8181dc46579` |
| Packed relative trial outcome | `3457e2531847c055823a5360d64b45bfaba19950799f6211aecd4fe016ef2b7f` |
| BF16 exponent survey harness | `5507b5bf1938660348b726caac31475842009b2423a01d4d9594cf43e28c0636` |
| BF16 exponent survey receipt | `433d5dafe5f28074b4466d7ed530ea3b6697a4e6440867fdf667f45f362f1c8e` |
| BF16 exponent screen executable | `8061ee7c66704b8dbb2b52373ed3dbf4ebb3e52d05c3836bfc67ddeb9979abfb` |
| BF16 exponent screen source patch | `2faca2b8d0c933f066093b08cd80a6a75d4cc82afbdf7cd154b45bd9c8ce4668` |
| BF16 exponent screen stock 4K | `75e032067f179b5d9c0f79fc395aa564ca827048d50b1c079e85f50ec11e1984` |
| Initial shared KV executable | `e68110ef2144a83597c73b6fa8e420c7d0845af737b46555aa1f5248f94a7ee0` |
| Initial shared KV patch | `e28cd0b92c0c15c58f64ba64c2dcbe32c3c28676cddb157075e9f2fb0983d28d` |
| Initial shared KV 4K receipt | `d21bbcbdf480d971fe981e44873e8ca77f8f91a17fe922c0589d9dbddf2e8080` |
| Fused expert and shared KV screen executable | `6e5978804c7360d9e0201f141241481a5cfd97b143ab85372c44d261551dbb54` |
| Fused expert and shared KV screen patch | `cb0b60a9153ca0ea989f8d223865bab9d9796431dbc702946a55e775eb9f3e3c` |
| Fused expert and shared KV short receipt | `6a537a37991c09b9869a5c779c0abc3ca126e3ae8fe566e66a76b9dbcab3af1b` |
| Fused expert and shared KV stock s4096-00 | `7e4e0ddbcfbeb05730eda56753b9eed6b161d0717b96a992e415f78980cf32cc` |
| Fused expert and shared KV stock g4096-01 | `7db6ef42c19e35f8310d33859e5936a0b13f5445fd44eb82ab60e447a6b1cca9` |
| Final shared KV and draft overlap source patch | `83f3ad64a6aa789d80ad8ecb59235935aa7f224325e7abe71493a4548ff86e1d` |
| Final shared KV and draft overlap diagnostic | `76daac573bfcb5bf438373d6e7e9fec5681bdd9538112ef319bfa16f61a779d3` |
| Final shared KV short | `c8c9d726979e9d63a191174e37e74c6a68366d926c5d1d7bf45197769908d74f` |
| Final shared KV stock g4096-01 | `01d2b254f7dde66895b9c411dc1b09585865c4c9842baa56fb680f7ff8b105af` |
| Draft overlap stock s4096-00 | `2b699f3aa4a6b4c7180575ba4cfcb9d0c03c8c9e346864ac7c5c76587a074b4f` |
| Draft overlap stock g4096-01 | `d1c8fa0a9cec82b7d1b09e6260b93ac8b04914f77bdbca1096f7b91e86611dd8` |
| Shared KV worker executable | `7e828c42c843236b99ba281bc81c65a089c34ddfdd59b2438a8205fd2122f25b` |
| Shared KV worker harness | `19cc3e7c6d692f44ab6e1227a68c9dd8e3dc373fafcf82a616d054e389cef1ab` |
| Shared KV worker pairs | `7ce148dbc61ef573e96b73330da18b9ee2880fdc2e8ff0b62fe858ab0010024c` |
| NVFP4 gate table executable | `c81f8e0da5b870b3b0a12895a6dc6e1f78418bfc3b1823d708fcfec81642308a` |
| NVFP4 gate table source patch | `2d007eccd9d77a43bcd534b7c993dd1f36d63e97f323eb1a95d99b53e01373d6` |
| NVFP4 gate table short | `90a12dcc58007a3298f1719a26b5ffeec084672f0d4b6f36bd44815ff5dd7e4a` |
| NVFP4 gate table stock s4096-00 | `0fe2f6ada86a075368723a8125834618653659bafe4cbe9b42a08af2ed0b8799` |
| NVFP4 gate table stock g4096-01 | `f2e87e67431a0ea27932e4565b7f8cc1132233031fbeefae5c47c58f8264c572` |
| Exact NVFP4 run width executable | `908ee728dabb251d11da51d385abc0467e0572a791fbcbce265495d13d0e5488` |
| Exact NVFP4 run width source patch | `3498b3e734b38329ca082f2d989021454ffa50c9a2726f09245701bf6c8ca574` |
| Exact NVFP4 run width short | `e9313d9075ac737c42a0dfde5e5a2ef7389d7ea0d19aada9bba4e3e654c598ca` |
| Exact NVFP4 run width stock s4096-00 | `708bdce09f890bc9b0eb0f308d3beb62aa28905633fb32763330f3a552e1bf1f` |
| Exact NVFP4 run width stock g4096-01 | `27bea16471d526e4a4650f6bad0218f7690bab17391867d642422635de12ff3c` |
| Exact NVFP4 run width worker | `46e1ba24b710f97bb836a5410f0e72b6d0fef39be861a923910eefe41edc1089` |
| Exact NVFP4 run width worker harness | `28c781711df27acd8d18f31cbb86c362ecd204dea8bc4f8c7abfdab03609b2e5` |
| Exact NVFP4 run width worker pairs | `48e6a8233b248b088012d57747225f9ff14d03da1fc479277d2e289680772876` |
| Exact NVFP4 run width flags-unset worker pairs | `5e24786a4a83638dda78cc3ae63205675c7ec669b1c68aae8d83554c632d69e7` |
| Flags-unset connected short | `3ec9bea159f2df8ab1c508329cd2b102cb53e672c740ae3fbddc9ba677cbfa63` |
| Flags-unset connected 4K | `c4a5cefe72c1633b5a56a7092e2e47479698d79936933e66910289b867c40652` |
| Flags-unset full target short | `4a5918ea804a28d00b444f2fa38323ec614e78551636b549db72a145476f2fa4` |
| Flags-unset full target stock 1 | `dd127f200ff7166b8161629ac330c933f81b66b3ab1d71fa8e23564206826e49` |
| Flags-unset full target stock 2 | `a19c16c494253bf1e80c9aa1b84a6411cdf03d0cf6c58943f04ddf57e6419cb3` |
| Flags-unset full target stock 3 | `6f9e4c7eba1ece95f4a06dd48424ab2d954c46842ead262b147f45b55de26ca1` |
| Flags-unset full target stock 4 | `b8258668677b64424f334f19d6341fc2f31f77281752f7019ed9f13175a8b04c` |
| Combined shared KV and exact run width stock s4096-00 | `78943fdcc69fd9eb3a7f453cd6a1acb0656b29cd0ac30b9b87d9c1f21c83460e` |
| Combined shared KV and exact run width stock g4096-01 | `36fb6feb61a8d7cb255400492a52cb34d07c838dc940f6ad3e611ad59474f32f` |
| Adaptive verifier worker | `1e262ee43ddb978e43f0229dfbeed7470bb59d134cd14095c8f4b65fa47f036f` |
| Adaptive verifier source patch | `70d058d1b86787b84a9a61fa8be2680769a42981bc5d89b89fe76da3863c1eef` |
| Adaptive verifier worker harness | `c118058a3ea084fc41ea5f2ef46769e418bccd93d27b9751489ef6cc319af300` |
| Adaptive verifier worker pairs | `fae03fab02edde52bfbea8ec1bd4b4cfb51b4b330cea0add5e02f25d0965d071` |
| Adaptive verifier host samples | `0d800841a554e080ce7e00ac928bfe7cbbd7380332546ecddb692c940dcbd9d0` |
| Verifier ingress and launch screen inkling_slice | `02e2197d67f63a349dca6e197637bdce243064666aea0d6d290b8a8238e6a9e6` |
| Verifier ingress and launch screen scored-source.patch | `c1d24465ae7cec006f61db72db3face3b3bd3fd6200c911e63d5d9abbe9d616d` |
| Verifier ingress and launch screen short.json | `1fb6a7b4c82ceced5fbdd6fee6b7227136147af0930f09665e861cc0df8f0858` |
| Verifier ingress and launch screen stock-s4096-00.json | `5012f7c1f4cf936255e811d75763fb823192be6c77506bf34c235a4a2645eb07` |
| Verifier ingress and launch screen stock-g4096-01.json | `170edc7fed7f2fa519147ccd3a6c3638b78399b770aacb153e7a6a612008a96a` |
| Draft shortlist and retained ingress native inkling_slice | `38285bf6047deb59935369d350ae8946259df37bc0b8e10f3b249aa6773958f6` |
| Draft shortlist and retained ingress native scored-source.patch | `51aaaa26772f64116b3ce08855a0638c68cd2c641cd08cbc343040964f481bf1` |
| Draft shortlist and retained ingress native short.json | `c1ccf02a260b214b4b72eae1b782fc4f021b227aa21136608a3e8ff6ffbfaad4` |
| Draft shortlist and retained ingress native stock-s4096-00.json | `71037a4d96fabdd24fc5d08a6282e3f465b04aa0e4b41a8e248983364d54e702` |
| Draft shortlist and retained ingress native stock-g4096-01.json | `e96ad68e9bb6e88170f44379f836f65365725b6dcfcfa112131acacee70318f6` |
| Adjacent output column screen inkling_slice | `d546261eb91f6229d270a925da4ae438290b45de780f785a58de0118dd69ce20` |
| Adjacent output column screen scored-source.patch | `64df71e683ca2138de006d389be2a03e3ac52052101d1fa68438650ef3e6f7e3` |
| Adjacent output column screen short.json | `daf12444f3a1a7c32d41c1b55ff1fdd4d893d3a03a101768b38a6f75644af7b3` |
| Adjacent output column screen stock-s4096-00.json | `af11508df29938a0ec7a10e527341e765498d46aa2c08cf53cd7cd6285d41217` |
| Adjacent output column screen stock-g4096-01.json | `baf968a8c447a986e3dabb3de0e3b5a9bed6fadd2c024dd780e7d04777c986cf` |
| Binary16 ingress and full shortlist worker | `60b9a9f16f597114e8cc8de53309aeb92acd0df82d0667e144ded8c31d051295` |
| Combined worker source patch | `93f798f65f4db89a5b3e921d29ad58842e51b781aa44158cbe6d33ca0352b672` |
| Combined worker harness | `b0151cac3deac14b5fb0bacfd44c426de88ad5651a28ca363f5e229f99fa513e` |
| Combined worker pairs | `b0f51cbf73dc71d42516807753c82f75ba1ab135370efb665809b0acb570a54b` |
| Combined worker host samples | `8cd2f651125ea6f008956de0e88950c80273d4f59e428e6038ddf0fb2168e50a` |
| Exact K16 TensorOps screen inkling_slice | `33675d717b8d25da655184b0e67c3928f9536a126ed27cd4b4cd8e260695b7f1` |
| Exact K16 TensorOps screen scored-source.patch | `3b36ab94cc13f2e5e893a799b8934696f55fe6e86f1de1fa9e69f01c2c261f26` |
| Exact K16 TensorOps screen short.json | `4feb786b94a6bf5ae199299f5b8b0865c28aa8a48859f94853841341c5f9c3a4` |
| Exact K16 TensorOps screen stock-s4096-00.json | `9df93b58a0edfde974fd01448fa643e252d7b111f6f6c3d40ba52e57d7835dc0` |
| Exact K16 TensorOps screen stock-g4096-01.json | `bd3d2ecb155ebf9efc63a875e31b1285257d8d81b8558b35e1299d2e5fee6685` |
| Final retained native executable | `2177af74dd54c72554c5665e921572dc888d4822dc44a27d005ad7929dd6a28c` |
| Final retained opt-in short | `16cafffd43c813483b5c399f258aa8ac65e037828da43a893580b8ad948f8991` |
| Final retained opt-in stock s4096-00 | `7b0d585aa002901e3d6b1f69f6d6cdfe7deb6968179479cf3b3d72d9558474b7` |
| Final retained opt-in stock g4096-01 | `6f48f18cf04c2ba0fe956a9fbd5c6a8289c2628ec1396c2d8808b6e54e35dbf1` |
| Final flags-unset connected short | `3ec9bea159f2df8ab1c508329cd2b102cb53e672c740ae3fbddc9ba677cbfa63` |
| Final flags-unset connected 4K | `c4a5cefe72c1633b5a56a7092e2e47479698d79936933e66910289b867c40652` |
| Final flags-unset full target short | `942f0e0f6d6b162e32ef6ad8b71dacccbe98fed7a797c060816a48d111ee6745` |
| Final flags-unset full target stock 1 | `25547ff564e1378aab815b3165979355fa640937700ebae0923852df128c3bf6` |
| Final flags-unset full target stock 2 | `d82d38427231954108d92f2039c1528ba842fa571586b9886d6399c05e1b30f8` |
| Final flags-unset full target stock 3 | `89ca16a9550f3d3c8c862e810de54fd19d5456b48c40118c2a18d1741db62975` |
| Final flags-unset full target stock 4 | `0217079a7bdb3b829de6f1bb84d1af2a3642dff7b49356865d567af7caf50ba5` |
| Short-context lookup script | `57222c0337f0ec31cbadc41692f33691bce7a048963ce8baaa7e526cdeadcde7` |
| Short-context lookup aggregate receipt | `b516de695a3540c8a2e05588d861c737a94258608d231dc8df60553d1c67dd83` |
| Full draft overlap with shortlist worker flag | `2b699f3aa4a6b4c7180575ba4cfcb9d0c03c8c9e346864ac7c5c76587a074b4f` |
| Compact run screen inkling_slice | `9bf391f5eab8921bfbee37ff48fde0c1b1d123dca908c3f8dbe97c97436ae775` |
| Compact run screen scored-source.patch | `cc06174fbd3f3a9751763d14e4dc5f4d6cdea60147f934fed2540fe9b818c45a` |
| Compact run screen stock-s4096-00.json | `60189baee7d353526f8f96828ebd26b5fffd4f6bc4a72b1ddceda63e609c4de6` |
| Compact run screen stock-g4096-01.json | `faef921b48a090da9de9627d9d3361604c302937146498e73f33e16080ed7993` |
| Compact run screen short.json | `3c186e3d5b29292dd5c4b1ed55416c2c8f68f452f0821860b58b2caebe78af84` |
| Expert finish screen inkling_slice | `8aa2b3e147192899d4e07f6b50d421618725210eeb4306520b2eab5f0650ac80` |
| Expert finish screen scored-source.patch | `a8208d6d1b2ad181c7cb12e4b5648714157daad417c30fb0c0b6620d8c0dff27` |
| Expert finish screen stock-s4096-00.json | `a9328de8f47ce488d2dbde3dc9e928ba96053f0003796befaa37e947100e1055` |
| Expert finish screen stock-g4096-01.json | `bf5104e70069af1e2c06c836cd8afc09f9ad101f31f74df044d6340dc39f091b` |
| Expert finish screen short.json | `4f93872ab817ddb67d3ca3869499c46f5b17bd431fee06537a390a21f6245f58` |
| Attention finish screen inkling_slice | `f9b653023056e4d63f656b5c11c210373ddc176864a6687b47a5cb873c9e7512` |
| Attention finish screen scored-source.patch | `d263b40a65f9e90c970b4682f636aafd2d0abbcbc99be6b38289f09c3c456ceb` |
| Attention finish screen stock-s4096-00.json | `22d0e3d5d39a159a6172453edf4ffa9f1c73a41e5b8529e02e4ec7315cb03115` |
| Attention finish screen stock-g4096-01.json | `aae77ec6396eb9adec50695d4f41b2668e38db5e723bbd95d0bc4287b94906f2` |
| Attention finish screen short.json | `eaf7885120fd232dadc7e0f3330306698c2e9c9e4a7e6e00c08f8e493870bd33` |
| Held-out calibrated sampling and sparse rejection inkling_slice | `8aa2b3e147192899d4e07f6b50d421618725210eeb4306520b2eab5f0650ac80` |
| Held-out calibrated sampling and sparse rejection scored-source.patch | `a8208d6d1b2ad181c7cb12e4b5648714157daad417c30fb0c0b6620d8c0dff27` |
| Held-out calibrated sampling and sparse rejection estimate.json | `256fac2014f1314659a0cb631eb7e4dbc945e3797cbbf8972bd72fae822a83a4` |
| Finishing fusions and compact runs worker | `3e6f69bb62329d6f27024a869d51dee6fce8014388c9444511a084f8d6eaa44e` |
| Finishing fusions worker source patch | `a4081a6dc60b7dde5f22ddde23957f4232b090f901f5a5cc75f583730d45424e` |
| Finishing fusions worker harness | `200e4c80262479f88ec53d9c8e2824df1eb7744e80bddbeefc00cfdc6a7c96ea` |
| Finishing fusions worker pairs | `f50386d0331ab37888050049165c7db30dfc279577173e9bc08a7628609e864d` |
| Finishing fusions worker host samples | `b6a7f1c88df9e44c717a99a9e1b0288bc4b775b95d527fd9b98f8dc7bed0894d` |
| Quantization threshold screen inkling_slice | `c1451c15fe19231ce4d607421efb8fc3deaca6c048496dd78f288b295238e33d` |
| Quantization threshold screen scored-source.patch | `a3d886e8ff293d7a44c898f50bbe059432f4ab2210002cfcbfc49a2de8e558e7` |
| Quantization threshold screen thresholds.py | `cf579cd72bb5228b14ff5d1cc8a14beac264669db8911b780fb3b699975b7481` |
| Quantization threshold screen thresholds.json | `53e6170415db9a74d1f7947d0df73144dd82205cf6b178d2b4ef663e136d2cde` |
| Quantization threshold screen short.json | `cd99c806a6b1a3d611554f1bb43c90a1e9c98fd8415b367a0a140efec348392f` |
| Quantization threshold screen stock-s4096-00.json | `f3072c9f2bd42bd92abba816c67433cb9eaaaf54289a079d79287d5c0f227c5c` |
| Quantization threshold screen stock-g4096-01.json | `537a6cf672181639857b407377ee3d77190c2b0713fc1d9e72e7990757cf2b3c` |
| Final compact runs and finishing proof inkling_slice | `4e63fb77b0015d8681f637074d9571c4076c6dd4252b9513a6c396b6049d57c5` |
| Final compact runs and finishing proof opt-in-short.json | `966c289930184cdc3a19c0183b316c538598fd631ab06f04c1e7ff4405581fc4` |
| Final compact runs and finishing proof opt-in-stock-s4096-00.json | `0561ce4c6b4d6f0089dd1540a49462022e82c95fdc40ff22b40af7e0304df5fa` |
| Final compact runs and finishing proof opt-in-stock-g4096-01.json | `11a4f37c98ecc9a8ca9243735ffe2a4132f2575abe4f19c46ff6ac5481963014` |
| Final compact runs and finishing proof flags-unset/slice-4.json | `3ec9bea159f2df8ab1c508329cd2b102cb53e672c740ae3fbddc9ba677cbfa63` |
| Final compact runs and finishing proof flags-unset/slice-4096.json | `c4a5cefe72c1633b5a56a7092e2e47479698d79936933e66910289b867c40652` |
| Final compact runs and finishing proof flags-unset/short-both.json | `a45201c551bd0a2952ebff682d0982cb8cf8503b9f425d54328bdd3611afe90b` |
| Final compact runs and finishing proof flags-unset/full-0.json | `93f68099b9ecbd4f0a839d6c49613c0808e009f0fd325e9c2936dce9fa212f31` |
| Final compact runs and finishing proof flags-unset/full-1.json | `d7a1abb9be1b88cc9f25734ac0115f23649fb3a2a2fc7a7e383143c01c07bd33` |
| Final compact runs and finishing proof flags-unset/full-2.json | `27cee79bbc18082d1b9d5a35b4c284c316af9969cda4278e50060ede9a11410c` |
| Final compact runs and finishing proof flags-unset/full-3.json | `2913aba71ce9b84ad35b5c0405484760cec07762b199789388ae872153a22711` |
