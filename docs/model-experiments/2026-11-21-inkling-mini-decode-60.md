# 20261121-INKDECODE-01–51 · Inkling 2 Mini exact decode toward 60 tok/s

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

## 20261121-INKDECODE-30 · Profile the live exact verifier transaction

Profile the production transaction command after restoring the same stock-scheduled 4K state. It includes undo copies, all target layers and exact F32 readout. It excludes the draft, CPU sampling and state restoration used to repeat a trial. Rotate four repetitions of the plain command, folded stage timestamps and expert-only timestamps at widths two through four, on two stock inputs. Every measured row matched serial F32 logit bits.

| Verifier rows | Plain wall median range across 4K inputs | Dispatches |
| --- | ---: | ---: |
| 2 | 28.38–28.43 ms | 2,349 |
| 3 | 32.54–32.99 ms | 2,827 |
| 4 | 35.75–38.13 ms | 3,353 |

The plain four-row GPU interval was 34.52–37.02 ms. Folded instrumentation measured 36.23–39.19 ms GPU, including 5.88–5.89 ms dense attention, 4.92 ms relative and cache attention, and 16.10–18.66 ms for experts. The separate expert-only instrumented arm attributed 9.22–10.95 ms to NVFP4 block products, about 3.20 ms to router projection, 0.45 ms to selection and bucketing, 1.00–1.02 ms to NVFP4 quantization and 0.37–0.38 ms to its epilogue. The markers perturb wall time, so use the plain result for transaction capacity and compare stage times within their own instrumentation arm.

Retain the diagnostic behind `STARK_INKLING_EXPERIMENT_PROFILE_VERIFY=1` with `--measure-verify-transaction`. Expert products remain the largest measured stage. Router projection has a measurable budget of its own, after previous work reduced exact route selection. No worker throughput or deployment claim follows from the profiler.

## 20261121-INKDECODE-31 · Share global K/V reads in the exact short verifier

Three query heads share each global K/V head. Vary how many of those heads and how many adjacent verifier rows consume each cache load. Keep the source score, relative table, sink, chronological updates and per-head shard reduction. Admit the shared path at positions of at least 1,024 with room for all rows in the global cache. The short prefix uses the ordinary path.

| Global sharing geometry | Two-row paired saving across 4K inputs | Three-row saving | Four-row saving |
| --- | ---: | ---: | ---: |
| Three heads, one row | 0.41–0.48 ms | 0.59–0.64 ms | 0.15–0.77 ms |
| One head, two rows | 0.28–0.35 ms | 0.34–0.50 ms | 0.64–0.86 ms |
| Three heads, two rows | 0.38–0.41 ms | 0.40–0.46 ms | 0.74–0.84 ms |

Each result is the median of eight rotated comparisons from the same stock-scheduled snapshot. Every checked F32 logit row matched the serial target. The three-head, one-row policy had the best three-row savings and passed all 45 rejection, capacity-wrap and rollback controls across one short and two 4K prefixes. It was selected for the first worker comparison because sampled depth two verifies three rows.

The sealed worker held all other verifier flags fixed and alternated four cold 4K pairs per mode, with greedy depth three and sampled depth two. Each request generated and emitted 256 tokens at context 4,352 with vision and DSpark resident. All 16 requests reused zero input tokens and matched the established exact output hashes.

| Mode | Control median | Shared global K/V median | Median paired gain | Candidate range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 53.54 tok/s | **54.19 tok/s** | +2.41% | 50.18–58.55 | 3/4 |
| Seeded sampled | 45.21 tok/s | **45.86 tok/s** | +2.20% | 44.62–47.47 | 4/4 |

Prefill varied across the paired session, with the slowest window affecting both arms. Use the same-session deltas for the kernel effect. Keep the shared global path opt-in with `STARK_INKLING_EXPERIMENT_SHARED_GLOBAL=1`. The measured rates remain below the 60 tok/s worker goal.

## 20261121-INKDECODE-32 · Vector-load the exact short verifier router

Reuse the vector-loaded singleton router for two through four live verifier rows. Keep the source 16 × 32 × 16 tensor operation, separate 2,048- and 256-channel accumulation, F32 output and existing route selection. Each aligned weight fragment loads four BF16 values together and only live input rows load a fragment. Limit the new entry to the short transactional verifier.

| Rows | Short paired saving | Two stock 4K paired savings | Faster 4K pairs |
| --- | ---: | ---: | ---: |
| 2 | 0.87 ms | 0.91–0.93 ms | 16/16 |
| 3 | 0.79 ms | 0.83–0.89 ms | 16/16 |
| 4 | 0.80 ms | 0.86–1.14 ms | 15/16 |

The eight alternating comparisons per width matched serial F32 readouts. The enabled candidate passed 45 rejection and capacity-wrap controls across one short and two stock 4K prefixes. The flags-unset build reproduced the connected four-token and 4K aggregate records. The full-model short control matched native state and pinned logits. Four stock-scheduled 4K cases matched 16 F32 logit vectors, 768 BF16 visit comparisons and four snapshot replays.

The latest global K/V worker pairs also bound the remaining seeded sampled work. Holding the observed round counts and all other time fixed, 60 tok/s would give the verifier 25.00–27.96 ms per round across the four prompts. That requires removing another 8.82–10.93 ms per round from the measured 33.82–36.93 ms. Calculate the budgets from the reported 255 decode intervals and each request's actual verifier time. The short-router saving covers only part of this gap.

The paired worker reused the global K/V configuration in both arms. It enabled only `STARK_INKLING_EXPERIMENT_VERIFY_SHORT_ROUTER=1` in the candidate. All 16 requests across four pairs per mode matched the pinned output hashes at 4,096 inputs, 256 outputs and zero reuse. The greedy and seeded sampled depths remained three and two.

| Mode | Control median | Vector router median | Median paired gain | Candidate range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 58.81 tok/s | **59.32 tok/s** | +0.78% | 51.89–61.66 | 3/4 |
| Seeded sampled | 46.62 tok/s | **47.70 tok/s** | +1.91% | 46.98–50.71 | 4/4 |

The workers accounted for 99.86% of observed GPU counter increments over 84 intervals. Thermal readings were nominal or fair. Pageouts increased by 1,225 pages and swapouts by zero. Keep the short verifier router opt-in. With the observed sampled round counts and other costs held fixed, the candidate still needs 6.23–8.73 ms less per round to reach 60 tok/s. One greedy candidate window exceeded 60 tok/s. Its paired control also exceeded 60.

### Flags-unset confirmation

The stock 4K profile observed `ink_singleton_router_source`, the promoted 64-row NVFP4 prefill tile and the MXFP8 tensor path. The singleton router's F32 logits, route IDs and weights exactly matched the original source router on the restored 4K state.

The rebuilt default worker, with drafting off and every Inkling flag unset, measured **34.12 tok/s greedy** and **32.47 tok/s seeded sampled** at the medians of two cold requests per mode. The ranges were 34.04–34.21 and 32.25–32.68 tok/s. All four output receipts matched their reference hashes at 256 outputs and zero input reuse. The first smoke harness applied the experimental verifier's three-draft readiness assertion to the stock worker and stopped before scoring. The corrected harness requires the stock worker's one-draft capability and checks the same output contract.

## 20261121-INKDECODE-33 · Stage NVFP4 ingress across wider output tiles

The four-column short verifier has four SIMD groups per threadgroup, with each SIMD group handling one output column. Compare eight and sixteen columns per tile, with each SIMD group completing columns in sequence. This keeps one set of per-route accumulators live. Test direct activation loads and a cooperative threadgroup copy of the live run's binary16 activation rows at each wider tile width. The up and down shaders use separate staging allocations of up to 18 and 9 KiB. All variants preserve K16 dot grouping, the source-lane scaled FMA recurrence and the final SIMD reduction.

| Four-row verifier variant | Paired wall increase across the two stock 4K inputs | NVFP4 product GPU medians |
| --- | ---: | ---: |
| Four columns, direct control | — | 9.31–11.44 ms |
| Eight columns, direct | 0.48–0.64 ms | 9.72–12.14 ms |
| Eight columns, staged | 1.35–1.71 ms | 10.75–13.21 ms |
| Sixteen columns, direct | 0.39–0.96 ms | 9.78–12.23 ms |
| Sixteen columns, staged | 0.79–0.91 ms | 10.31–12.85 ms |

The wall comparisons use six rotated same-state pairs and the GPU totals use two separately instrumented pairs. The plain four-row controls took 37.16–38.10 ms. All variants reproduced serial F32 readouts at widths two through four on one short and two pinned 4K prefixes. The enabled eight-column staged arm passed all 45 rejection and wrap controls. Wider direct tiles and staged tiles also lost at widths two and three on both 4K inputs. The four-path comparisons separate the effect of staging from the cost of changing tile geometry.

Retire the candidate. Fewer device requests for activation rows did not reduce whole-command wall time in these layouts. The screen measured GPU stage time, so it leaves the balance between cache hits, scratch occupancy and reduced grid parallelism unresolved. No worker speedup claim follows.

## 20261121-INKDECODE-34 · Bound adaptive depth on exact seeded trajectories

Start four sampled teacher trajectories from pinned stock 4K logits. At each of 192 anchors per input, run the actual full-vocabulary shortlist proposer separately at depths one, two and three. Time each draft depth in rotated order. Compare each proposal prefix against the same exact temperature-one teacher continuation. In the replay, advance by the accepted prefix plus one so overlapping draft opportunities cannot count twice.

Charge a favorable 22.80 ms for an ordinary step. The measured ordinary medians in the final pass were 23.19–23.58 ms. Charge the fastest same-state plain verifier median from the wider-tile control, 27.47, 31.68 or 37.16 ms, plus the input's measured draft median at the chosen depth. The model leaves out finishing and host costs. Its rates are estimates on the recorded trajectories.

| Stock 4K input, sampled | Fixed depth two | Best fixed depth per input | Anchor confidence | Future draws, ratio | Future draws, 16-token window | Hindsight optimum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s4096-00` | 51.84 | 51.84 | 51.84 | 52.01 | 50.99 | 68.65 |
| `s4096-01` | 59.42 | 59.42 | 59.42 | 59.42 | 59.12 | 74.39 |
| `g4096-01` | 63.62 | 63.62 | 61.41 | 63.50 | 63.24 | 77.13 |
| `g4096-02` | 48.03 | 53.92 | 42.68 | 45.80 | 45.87 | 60.85 |

Rates are in modeled tok/s. Fixed depth two was the best fixed setting at the median across inputs, 55.63 tok/s. Anchor confidence uses three bins of the already-sampled anchor's top-two target logit gap and whether the draw chose the top token. The future-draw arms use a copy of the local sampler state to place the next draws into eight bins. Fit the conditional proposal acceptance by depth, mask and draw bin on the other three inputs, with smoothing toward the same-mask mean. Choose depth by estimated cost per emitted token or minimize expected cost over 16 positions. The original sampler and draw sequence are unchanged by the replay. The hindsight arm uses future acceptance outcomes to compute the lowest cost among all valid depth schedules and quantifies the value of a perfect predictor at these fixed costs.

The ratio policy estimates a median of 55.72 tok/s, about 0.09 more than fixed depth two, and falls behind on the hardest input. The 16-position policy estimates 55.05. Anchor confidence alone failed to improve on fixed depth two. The hindsight range of 60.85–77.13 shows that perfect round choice could matter, with only 0.85 tok/s of modeled headroom on the hardest prompt before finishing and host work. Defer implementing these predictors. The screen supports lowering per-round compute or improving the proposals themselves before revisiting depth choice. No worker throughput changed.

## 20261121-INKDECODE-35 · Isolate packed expert reads and simplify exact coefficient decoding

Run three shadow projections from the live short verifier's routed operands using the same packed banks, indirect grid and scratch write layout. Time the original dot, the original dot with one weight column reused across each expert's outputs, and an integer checksum that consumes every packed weight, scale and binary16 activation load. Each shadow writes private scratch before the unchanged live product. Track the live product separately and compare its F32 logits against serial. This measures the response to address reuse and removing float work under these launch conditions. The probe does not identify DRAM traffic or decompose wall time into additive costs.

In six rotated shadow comparisons per width on each stock input, the four-row shadow control took 11.93–12.21 ms for the 88 gate/up and down products. Reusing weight addresses saved 0.06–0.45 ms at the paired medians. The checksum saved 4.83–5.57 ms with the reads still present. The live projection medians moved by at most 0.26 ms relative to their same-input shadow baseline. The result favors addressing arithmetic and coefficient decode before redesigning packed-weight storage.

Form the FP4 magnitude as an affine FP32 bit pattern for codes two through seven, with explicit code-one correction and a zero mask. Restore the sign bit after the mask. For E4M3, separately test shifting the complete byte into adjacent normal exponent and mantissa fields, with an explicit subnormal select. The host bit comparison covered all 16 FP4 nibbles, including negative zero, and all 256 inputs to the unsigned E4M3 helper. The signed MXFP8 reserved-code handling remains independent.

| Exact decoder arm | Two-row paired saving, stock 4K | Three-row paired saving | Four-row paired saving |
| --- | ---: | ---: | ---: |
| FP4 affine bits | 1.14–1.31 ms | 1.56–1.69 ms | 2.06–2.30 ms |
| E4M3 linear bits and subnormal select | 0.00–0.24 ms | −0.17–0.18 ms | 0.33–0.40 ms |
| Both | 1.21–1.31 ms | 1.57–2.07 ms | 2.07–2.25 ms |

Use eight rotated uninstrumented pairs per width and input. Separate timestamped comparisons put the four-row NVFP4 products at 11.78–12.43 ms with the ordinary decoder and 9.66–10.65 ms with affine FP4. All checked F32 vectors matched the serial target on the short prefix and both 4K inputs. The combined candidate passed the 45 rejected-suffix and capacity-wrap checks. An initial Rust build failed to resolve the private scratch allocation helper. The sealed run uses a directly allocated Metal private buffer.

## 20261121-INKDECODE-36 · Compare exact FP16 element products

The ingress coefficient already contains one FP4 value multiplied by its E4M3 activation scale. A second FP4 value leaves an odd significand no larger than `3 × 3 × 15 = 135`. Each nonzero element product is within the normal FP16 exponent range for the source activation codes. Multiply the two binary16 values, widen each product before the four-element sums, and preserve the F32 weight-scale FMA recurrence and final SIMD reduction. A host bit check covered 32,768 combinations of activation nibble, unsigned E4M3 scale and weight nibble, including signed zeros.

| Two-stock-input verifier arm | Two-row paired saving | Three-row paired saving | Four-row paired saving |
| --- | ---: | ---: | ---: |
| Exact FP16 element products | 0.78–0.89 ms | 0.96–1.12 ms | 1.25–1.46 ms |
| Affine FP4 readout in F32 | 1.01–1.18 ms | 1.36–1.44 ms | 1.68–1.94 ms |
| Affine FP4 plus linear E4M3 in F32 | 1.09–1.19 ms | 1.36–1.52 ms | 1.63–1.95 ms |

Eight rotated plain comparisons at each width reproduced every checked F32 row on the short and both stock 4K fixtures. Exact FP16 element products also passed all 45 rejection and capacity-wrap controls. The affine change alone beat FP16 multiplication on both 4K inputs. Its repeated four-row command savings were 1.68–1.94 ms. The linear E4M3 change gave mixed standalone results and little further saving in the repeated comparison. Retire the half-product, E4M3 and shadow variants. Retain the affine FP4 decoder within binary16-ingress verification under `STARK_INKLING_EXPERIMENT_VERIFY_LEAN_DECODE=1`.

### Retained-build parity

The build with affine FP4 decoding reproduced the serial continuation and committed cache state with the candidate enabled on the short and both stock 4K captures. All 45 rejection and capacity-wrap controls passed, including abandoned-transaction reset. The flags-unset source matched both connected slice aggregate records and the pinned full-model short logits. Four stock-scheduled 4K captures reproduced 16 F32 logit vectors, 768 BF16 visit residuals and four snapshot replays.

### Paired cold 4K worker results

Both arms used the same sealed worker and kept the prior exact verifier, global K/V sharing and short router options on. The candidate additionally enabled the affine FP4 decoder. Four alternating pairs per mode used 4,096 input tokens at context 4,352, zero reuse and 256 sampled and emitted tokens. Vision and DSpark stayed resident. Greedy requested three drafts and seeded sampled requested two. All 16 output hashes and paired proposal and acceptance counts matched.

| Mode | Control median | Affine FP4 median | Median paired gain | Candidate range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 57.63 tok/s | **60.82 tok/s** | +5.57% | 52.71–64.23 | 4/4 |
| Seeded sampled | 46.03 tok/s | **49.22 tok/s** | +5.62% | 47.91–49.28 | 4/4 |

The two workers accounted for 99.87% of observed GPU counter increments across 84 intervals. Thermal readings were nominal or fair. Pageouts increased by 286 pages and swapouts by zero. The greedy median clears 60 tok/s on this set. At the observed sampled acceptance and other costs, reaching 60 would still require 6.78–8.00 ms less per verifier round across the sampled prompts.

## 20261121-INKDECODE-37 · Profile the stock 4K DSpark proposal command

Measure the live multi-mask proposal command at depths two and three on two stock 4K prefixes. Alternate the plain and timestamped arms at each of 16 consecutive anchors, and advance the same source target teacher once per anchor. Match all proposed IDs between arms. The full-vocabulary shortlist is enabled. The timing excludes target verification, acceptance and commit.

| Proposals | Plain command wall median | Eight-row draft block GPU | Shared vocabulary projection GPU | Markov preparation and state GPU | Rank and winner GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two | 4.62–4.64 ms | 2.12–2.13 ms | 2.06–2.07 ms | 0.14–0.15 ms | 0.06 ms |
| Three | 4.71–4.74 ms | 2.12–2.13 ms | 2.06–2.07 ms | 0.20–0.22 ms | 0.08–0.09 ms |

Plain command GPU medians were 4.37–4.40 ms at depth two and 4.45–4.48 ms at depth three. The extra timestamps added 0.01–0.02 ms at the paired wall medians. The eight-row block includes the final normalization before the vocabulary projection. The fixed draft-block and vocabulary costs explain most of the proposal command. The shortlist rank stage has less than 0.09 ms left to recover on these inputs. Keep the timestamps in the archived diagnostic patch. This measurement does not change worker throughput.

## 20261121-INKDECODE-38 · Shrink and widen the DSpark vocabulary projection tile

Keep the stock eight-row bidirectional draft, live masks and BF16 weights. Change only the shared vocabulary projection. Compare 8 × 32, 8 × 64, 8 × 128, 16 × 64 and 16 × 128 against the stock 16 × 32 tile. Use twelve rotated plain pairs and four timestamped pairs at each of two and three proposals on both stock 4K prefixes. Copy the private live F32 base scores after each timed command for the bit check.

All five shapes compiled and matched 160,051,200 F32 scores at the live rows, including vocabulary padding, and every proposed ID between arms. The 8 × 128 tile had the strongest wall results. At two proposals it saved 0.02–0.04 ms at the paired median, and at three it saved 0.04–0.05 ms. Instrumented vocabulary time fell from 2.03–2.07 ms to 2.00–2.02 ms. The kernel still reads the same weight bytes with fewer padded output rows and fewer output tiles.

The finer timestamps also split the original draft backbone. Attention parts and join together used about 0.34 ms, and feed-forward projections, activation and residual update used 1.14–1.16 ms across the three layers. Retire the vocabulary tile variants because the whole-command saving is small. Keep the exact stock geometry.

## 20261121-INKDECODE-39 · Encode the verifier while the draft command runs

Submit the same DSpark command first. While it runs, the CPU encodes and submits the target verifier to the same Metal queue. The anchor embedding still takes its ID from the CPU. Each proposed row reads its ID from the first command's winner buffer and uses the same embedding reduction and BF16 rounding. The queue preserves the dependency on draft completion. The target command retains the original undo copies, row order and arithmetic. Verify both command statuses and the draft ID extent before exposing the transaction. If either command fails or a proposal ID is invalid, wait for the queued work and reset the session. The embedding shader bounds its read even on the invalid-ID path.

| Proposals | Short paired wall saving | Stock 4K paired savings | 4K wrap controls, paired savings |
| --- | ---: | ---: | ---: |
| Two | 1.38 ms | 1.49–3.03 ms | 1.16–2.06 ms |
| Three | 1.96 ms | 0.24–2.72 ms | 1.06–1.67 ms |

The timing boundary includes the draft and verifier commands through completion and readback. Restore the same snapshot before each arm and rotate command order. The 48 normal native pairs and 32 capacity-crossing pairs matched proposals, every provisional F32 readout and committed target and draft state. Each width exercises every nonempty accepted prefix. The queued path also reproduced the full 96-token serial continuation at each depth and all abandoned-transaction resets. The usual 45 rejection and capacity-wrap checks passed in the same runs.

The queued path is admitted for two or three proposals in source text mode under `STARK_INKLING_EXPERIMENT_VERIFY_OVERLAP=1`. Single-proposal and final-tail rounds use the existing calls. The timed split assigns the pre-verifier wall interval to draft and the rest to verification. In this mode the verification interval can include draft work still running on the GPU. Use the combined interval for cross-arm comparisons.

The flags-unset executable reproduced the pinned connected short and 4K slices. The full-model short logits matched, and four stock 4K captures reproduced the 16 F32 vectors, 768 BF16 visit residuals and four snapshot replays. The input cache and committed state format used by prefix reuse are the same in both queued and ordinary transactions.

### Cold worker screen

The two arms used the same sealed worker and baseline affine FP4 verifier settings. Only the candidate selected the queued path. Compare four alternating cold 4K pairs per mode at context 4,352, zero input reuse and 256 sampled and emitted tokens. Vision and DSpark stayed resident. Greedy used three proposals and seeded sampled used two. All 16 generated output hashes and paired proposal and acceptance totals matched.

| Mode | Control median | Queued median | Median paired gain | Candidate range | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 60.60 tok/s | 61.16 tok/s | +1.00% | 52.15–64.17 | 4/4 |
| Seeded sampled | 48.39 tok/s | 48.89 tok/s | +0.40% | 46.85–51.25 | 2/4 |

The workers accounted for 99.86% of observed GPU counter increments over 85 intervals. Host readings were nominal or fair, with 509 pageouts and zero swapouts. This screen shows a small greedy effect and mixed sampled results. The stronger two-proposal native savings did not transfer consistently to useful sampled decode.

### Sampled repeat with explicit overlap counts

The final worker adds `overlapped_rounds` to distinguish measurements whose timing boundaries include pending draft work. Repeat all four seeded sampled pairs with the same scope. Each candidate receipt counted exactly its depth-two and depth-three rounds, and the ordinary receipts omitted the field. All eight outputs and paired proposal and acceptance totals matched.

The repeat moved the median from 50.93 to **52.87 tok/s** and improved all four pairs. The median paired gain was **3.82%**, and candidate rates ranged from 51.58 to 56.60 tok/s. The combined proposal and verifier interval dropped by 1.37 ms per round at the paired median. The earlier sampled screen saved 0.15 ms at that boundary. The repeat's two workers accounted for 99.86% of measured GPU counter increments over 41 intervals, with fair thermal readings, 25 pageouts and zero swapouts. Keep the queued mode opt-in given the variation across sessions. At the observed acceptance and other costs in the repeat, sampled decode still needs 2.04–5.07 ms less per round across the four inputs to reach 60 tok/s.

The final worker reused 2,048 aligned input tokens across source draft depths and matched the cold output. It cleared retained prefix state on decode cancellation and when switching between source drafting and ordinary generation. The opt-in kept the same short and long-context admission decisions. An early stop on an accepted proposal matched the ordinary continuation and preserved sampled-token accounting.

## 20261121-INKDECODE-40 · Compare one command buffer with queued verification

Keep the GPU-resident draft IDs and the original target rollback. For the one-buffer arm, save the undo slots, encode the unchanged draft block and append the verifier dispatches to the same command. The queued arm first submits the draft and encodes the dependent verifier while the GPU is running. The third arm completes the draft, reads its IDs on the host and then encodes verification. Compare eight rotated observations for two and three proposals on the same restored short and two stock 4K states. Repeat both depths at the capacity-crossing states. Each mode matched the proposals, every provisional F32 vector and the committed state for every accepted prefix tested. The one-buffer mode also matched the serial 96-token continuation and abandoned-transaction reset.

| Proposals | One-buffer wall increase over queued, short | One-buffer wall increase over queued, stock 4K | 4K capacity-crossing increase |
| --- | ---: | ---: | ---: |
| Two | 0.93 ms | 0.38–1.80 ms | 0.86–1.75 ms |
| Three | 0.99 ms | 1.22–1.29 ms | 0.95–2.42 ms |

The boundary includes proposal and verifier encode, submission, completion and readback before acceptance. The one-buffer path has fewer submits and encodes both stages before starting GPU work. It lost to the existing queue schedule on both 4K states at both depths. Retire the one-buffer candidate and keep the queued worker path. No worker speed claim follows from the native comparison.

## 20261121-INKDECODE-41 · Specialize dense GEMV to the live verifier width

The existing shared dense kernel accumulates four independent dots at widths two and three. The unused inputs alias the first input, and the runtime width guards the stores. Compile two- and three-row variants using the same weight loads, K grouping, FP32 accumulation, SIMD reduction and BF16 output. Select the specialization only in the short target verifier. At width four both arms retain the same kernel and serve as a noise control.

| Verification width | Plain paired saving across stock 4K inputs | Dense-attention GPU saving across stock 4K inputs |
| --- | ---: | ---: |
| Two | 0.30–0.38 ms | 0.16–0.23 ms |
| Three | 0.14–0.18 ms | 0.08–0.17 ms |
| Four, unchanged kernel | −0.22 to +0.12 ms | −0.14 to −0.07 ms |

Each fixture used eight alternating plain comparisons and four separate timestamped comparisons per width. All checked F32 logits matched serial at short and 4K. The enabled candidate also passed all 45 rejection and wrap controls and the queued continuation and committed-state comparisons. The short whole-command effect was mixed. The three-row saving is below 0.2 ms at the usual sampled depth and close to the unchanged control's variation. Retire the specialization and keep the common dense path.

## 20261121-INKDECODE-42 · Measure the raw K32 NVFP4 tensor budget

Before building a certificate and replay scheduler, time raw TensorOps over the real sorted short-verifier runs. Decode the packed NVFP4 weights to exact binary16 in K32 threadgroup tiles and gather the original binary16 activation rows. Compare two SIMD groups over 64 output columns with M8 and M16 tensor fragments. Test continuous FP32 accumulation and independent K32 partials merged with TwoSum compensation. Run each shadow before the unchanged live expert dot, with the same weights, inputs, active routes and private scratch. Rerun the existing vector dot into that scratch as the control. Six rotated timestamped comparisons per width and input matched all live serial F32 readouts. Neither tensor result feeds the model.

| Shadow GPU time across 88 routed products | Three verifier rows, stock 4K | Four verifier rows, stock 4K |
| --- | ---: | ---: |
| Current affine vector dot | 7.29–7.74 ms | 8.18–9.86 ms |
| M8 continuous K32 | 15.67–16.89 ms | 17.58–21.87 ms |
| M8 compensated K32 | 16.87–18.25 ms | 19.11–23.29 ms |
| M16 continuous K32 | 15.81–16.82 ms | 18.12–22.92 ms |
| M16 compensated K32 | 18.07–19.03 ms | 20.09–24.77 ms |

Both geometries compiled. The fastest raw tensor tile took over twice the vector control's time at three rows before any BF16 certificate or replay. The live vector products after the shadow remained within about 0.27 ms of the corresponding same-input vector shadow arm's live medians. Retire this staged TensorOps layout before implementing replay. These measurements cover the tiled decode, synchronization, multiplication and scratch stores as one stage. A different input layout would need a new component screen.

## 20261121-INKDECODE-43 · Fold K16 dot reductions and specialize the lane loop

Use the affine half-ingress verifier as control. Form each K16 inner product with four vertical F32 FMA accumulators and one horizontal reduction. The K16 products and all partials are exact on the common activation-scale lattice. Keep the outer E4M3 weight-scale FMA and final per-lane SIMD reduction. Test the fold alone, fully unrolling the static number of K16 blocks per lane with a guarded final block, and both changes together.

| Variant | Three-row plain paired change, stock 4K | Four-row plain paired change, stock 4K |
| --- | ---: | ---: |
| Fold the K16 reduction | 0.07–0.36 ms faster | 0.03–0.07 ms slower |
| Unroll the lane loop | 1.97–2.02 ms slower | 2.54–2.88 ms slower |
| Fold and unroll | 1.87–2.02 ms slower | 2.61–2.69 ms slower |

Use eight rotated plain and two separate timestamped comparisons per width and fixture. Folding alone removed only 0.01–0.08 ms from the three-row native NVFP4 dot stage. The four-row product medians rose by 0.14–0.18 ms. All checked F32 rows matched serial on the short and stock 4K inputs. The selected fold arm passed all 45 wrap and rejection checks and reproduced the queued serial continuation. Retire both changes. The folded dot gives little component saving, and static unrolling materially increases GPU time.

## 20261121-INKDECODE-44 · Time register-fed K32 NVFP4 tensor tiles

Remove the staging arrays and per-tile threadgroup barriers from attempt 42. Keep two SIMD groups per 64 output columns. Fill each lane's cooperative activation and weight fragments directly from the live sorted routes and packed source tensors. The weight path unpacks each requested FP4 coefficient and applies its E4M3 scale before the exact binary16 conversion. Compare continuous accumulation and independent K32 partials with compensation. The eight-row cooperative pipeline was unavailable in this build. The sixteen-row pipeline ran both variants.

| Shadow GPU time across 88 products | Three verifier rows, stock 4K | Four verifier rows, stock 4K |
| --- | ---: | ---: |
| Affine vector control | 7.10–7.37 ms | 8.00–9.49 ms |
| Register-fed M16 continuous K32 | 61.67–61.75 ms | 69.54–79.88 ms |
| Register-fed M16 compensated K32 | 59.78–60.21 ms | 69.39–80.80 ms |

Six rotated comparisons per width and stock input preserved all checked live F32 logits. The shadow output stays in scratch. At three rows, direct register unpacking took more than eight times the vector control before the certificate or any exact replay. Retire this layout. Both the staged and direct-register screens leave the existing affine vector worker as the strongest measured path. The queued worker's 52.87 tok/s sampled repeat remains the end-to-end evidence for the current branch.

## 20261121-INKDECODE-45 · Replay exact seeded draft branches

Replay 192 anchors in each of four original-checkpoint stock 4K trajectories. Confirm the first complete F32 vocabulary row against the pinned oracle and produce the teacher continuation with the source-precision serial target and its original sampler draws. At each anchor, run the native one-, two-, and three-mask proposer. Also inspect the full first-mask score row for the two highest-ranked tokens distinct from the production two-mask choice. A fork receives credit only for the matching root and any verified primary child. Each step emits one fresh target draw and consumes the same seeded sequence.

Charge equal per-round costs to chains and forks at the same verifier width. Use representative costs of 23.5, 32.0, 36.0, and 40.8 ms at depths zero through three. Also test a favorable schedule of 22.8, 30.0, 34.5, and 38.5 ms. Both schedules omit tree bookkeeping and assume it is possible to verify the added rows with the existing width costs. These are replay estimates.

| Proposal policy | Representative median, tok/s | Favorable median, tok/s |
| --- | ---: | ---: |
| Best fixed chain | 56.17 | 58.70 |
| Two roots | 48.59 | 50.70 |
| Three roots | 45.13 | 47.82 |
| Two roots and one primary child | 52.61 | 55.76 |
| Perfect alternate root and one primary child | 59.18 | 62.71 |

With real alternate roots and a primary child, even an extra branch row at zero cost gives 59.52 tok/s at representative costs. A 1 ms added row gives 57.93, and a 2 ms row gives 56.43. A hindsight choice among fixed chains reaches 71.65 tok/s. Giving hindsight the measured alternate roots adds 1.85 tok/s. Retire this ranked fork before building branched cache and rollback support. The results favor reducing verifier cost or improving conditional acceptance at the existing width.

## 20261121-INKDECODE-46 · Quantize the multistep DSpark vocabulary head

Pack a separate copy of the complete vocabulary matrix with one symmetric scale per token row. Use the highest absolute BF16 coefficient divided by 127 for INT8 or seven for signed packed INT4, with scale one for zero padding. Feed the mixed Metal TensorOps matmul the same BF16 draft activations and accumulate in F32. Scale the live rows before Markov reranking. The projection shares the draft's normalized inputs and scratch. It uses the original BF16 head for one proposal and retains the exact target projection and sampler for every verification row.

Replay the four stock 4K trajectories and seeded teacher continuations from attempt 45. Rotate the BF16, INT8 and INT4 arms at each of 192 anchors per input and depth. Time the whole ordinary proposal, including Metal command submission and completion, after eight warm-up anchors. Check each proposal vector against the BF16 control and count the length of its matching prefix against the exact target continuation. Reuse the representative verifier costs from attempt 45, adjusted by each variant's observed draft-time difference, to estimate the effect of changed acceptance.

| Depth and metric | BF16 | INT8 | INT4 |
| --- | ---: | ---: | ---: |
| Two proposals, wall median range | 4.65–5.15 ms | 3.61–3.96 ms | 3.13–3.35 ms |
| Two proposals, matching complete proposal vectors | 768 | 758 | 553 |
| Two proposals, accepted teacher tokens | 841 | 840 | 818 |
| Two proposals, modeled median tok/s | 55.93 | 57.21 | 56.92 |
| Three proposals, wall median range | 4.73–5.22 ms | 3.69–4.03 ms | 3.20–3.43 ms |
| Three proposals, matching complete proposal vectors | 768 | 755 | 387 |
| Three proposals, accepted teacher tokens | 1,028 | 1,027 | 996 |
| Three proposals, modeled median tok/s | 55.37 | 56.80 | 54.93 |

The packed INT4 arm saved an extra 0.45–0.61 ms per proposal command, with lower modeled throughput than INT8 at both depths. Retire the per-row INT4 layout. Retain the INT8 candidate for the paired worker gate. It adds 461,747,712 bytes (440.36 MiB) of immutable weights and scales to the existing DSpark memory admission. Prefix snapshots continue to hold the same mutable state.

### Paired cold 4K worker results

Compare the queued BF16 control with the queued INT8 proposal head in the same worker build. Both arms use the affine NVFP4 verifier and full-vocabulary shortlist. Vision and DSpark stay resident. Each of four alternating pairs per mode uses 4,096 cold input tokens, context 4,352, zero reuse and exactly 256 sampled and emitted tokens. The seeded sampled requests use two proposals with temperature and top-p both set to one. Greedy uses three proposals.

| Mode | BF16 control median | INT8 median | INT8 range | Median paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seeded sampled | 50.82 tok/s | **53.40 tok/s** | 52.12–54.55 | +6.04% | 4/4 |
| Greedy | 62.89 tok/s | **61.86 tok/s** | 54.43–67.97 | +2.59% | 3/4 |

All 16 outputs match the pinned worker hashes. Sampled rounds total 531 with BF16 and 530 with INT8. Greedy rounds total 397 and 398. The two workers account for 99.86% of observed Metal counter increments across 81 intervals. Thermal samples report nominal or fair, with zero additional pageouts and swapouts. Median prefill rates were 898.61 and 914.21 tok/s in the sampled pairs, and 940.47 and 929.15 in the greedy pairs. These paired runs load two workers and enable the experimental verifier stack.

An initial worker build screened the same math before the admission revision. Sampled moved from 47.56 to 52.12 tok/s and gained 9.93% at the paired median. Greedy moved from 60.82 to 55.93 tok/s with a −0.95% median paired change and two faster pairs. That build charged the INT8 bank at per-request admission after allocating it at startup. The repeat uses the revised startup gate and leaves the teacher score export on BF16. Greedy timing varied by session. The sampled repeat still needs 3.40–4.79 ms less per round on the four requests to reach 60 tok/s at their observed round counts.

### Short and stock 4K parity

With flags unset, the connected short and 4K slice receipts matched the qualified controls. The four-token full-target run had zero unequal pinned F32 values. Four stock 4K fixtures matched all 16 checked F32 vectors and 768 BF16 visit-residual blocks, with four exact snapshot replays.

With the INT8 head enabled, the native verifier matched its serial continuation and complete committed state through 96 outputs at each of proposal depths one, two and three, on the short input and two stock 4K fixtures. All 45 rejected-suffix and capacity-wrap controls passed, along with abandoned-command reset and cross-mode snapshot rejection. Each diagnostic run recorded 461,747,712 bytes of INT8 head and scale storage.

With flags unset, the same build completed four cold 4K requests per mode with the exact one-row router and matching output hashes. Useful decode medians were **33.89 tok/s greedy** and **32.20 tok/s seeded sampled**. Prefill medians were **1,048.15 tok/s greedy** and **1,050.30 tok/s seeded sampled**.

The INT8 worker reused a 2,048-token prefix across source proposal depths with exact continuation. It cleared the prefix on cancellation and when switching between source drafting and ordinary generation. The applicable short and long-context requests still reject three drafts, and an early stop on an accepted proposal preserved the exact output and sampled-token count. The first lifecycle harness was rejected before generation because its private image-check directory lacked owner-only permissions. The corrected harness completed all checks.

Keep the INT8 head opt-in under `STARK_INKLING_EXPERIMENT_DRAFT_INT8=1` with the admitted source verifier. Charge the optional bank at startup and its actual resident bytes during request admission. The idle worker's Metal allocation increased by 461,750,272 bytes, within 2,560 bytes of the expected bank size. The BF16 proposal head remains the default and supplies the existing raw teacher-score diagnostic.

## 20261121-INKDECODE-47 · Check exact integer NVFP4 block arithmetic

The unsigned E4M3 scales are constant within each source K16 activation and weight block. Twice each decoded FP4 coefficient fits in signed INT8. A K16 dot of the doubled coefficients fits in signed INT32. Combine the activation and weight scales, divide by four and use the integer dot in the original per-lane F32 FMA recurrence. Preserve block and lane reduction order.

The local Metal 4 compiler accepts a BF16 × INT8 → F32 vocabulary projection and a separate typed INT8 × INT8 → INT32 K16 product. A host calculation matched 3,806,444 F32 block FMA controls across 59 signed-code patterns, all 127 finite unsigned scale codes for each operand and four prior accumulators. The direct integer screen used explicit Metal `int8_t` after the equivalent `char` spelling failed template dispatch.

Leave the integer expert path open. A useful GPU screen must include packing or repacking, the scale recurrence and real route occupancy before considering a new represented weight bank. This attempt establishes compiler and scalar arithmetic feasibility only.

## 20261121-INKDECODE-48 · Time exact packed integer expert products

Run the integer arithmetic in the live short verifier, including activation packing, on its original packed weight banks and sorted routes. All candidates keep the four-over-six activation error comparison. They store unscaled signed activation coefficients and one block scale. The normalized FP16 arm uses four exact half-vector subdots. The INT32 arm multiplies the doubled FP4 integers directly. The packed-pair arm joins two signed coefficients with radix 1,024 and extracts two products from the signed middle ten bits of one multiply. Every arm preserves the source's per-lane F32 FMA and final SIMD reduction.

| Candidate | Three-row paired wall change, stock 4K | Four-row paired wall change, stock 4K |
| --- | ---: | ---: |
| Normalized FP16 subdots | +2.67–2.72 ms | +3.51–3.59 ms |
| INT32 vector arithmetic | +3.71–3.78 ms | +5.02–5.42 ms |
| Packed signed pairs | +0.58–0.60 ms | +0.77–0.85 ms |

Use eight rotated plain comparisons and three separately timestamped comparisons at each width. All 924 checked F32 readout vectors matched the serial target across the short and two stock 4K fixtures. At three rows, the 88 NVFP4 products used 6.82–7.11 ms with the affine control and 7.42–7.66 ms with packed pairs. Packing slightly reduced the measured ingress time, from 0.99–1.00 ms to 0.97 ms. The first Rust build rejected an overbroad re-export, and the first Metal load rejected a scalar shifted by vector counts. The sealed run scopes the helper and broadcasts the lookup constant explicitly. Retire the integer variants and preserve the affine vector path.

## 20261121-INKDECODE-49 · Fuse and quantize the multistep DSpark feed-forward projections

Keep the admitted INT8 vocabulary head in all three arms. Compare the original BF16 feed-forward block, fused BF16 gate/up, and fused INT8 gate/up with INT8 down. The gate and up tiles write a combined output. Their epilogue rescales INT8 results, rounds both projections to BF16 and applies the existing SwiGLU. The INT8 down epilogue also preserves the BF16 cast before the F32 residual update. Quantize each projection row independently at load time. Attention, the one-proposal path, raw teacher-score export and the exact target are outside the candidate.

Repeat the four pinned seeded 4K trajectories at 192 anchors per input. Rotate the three arms at each anchor at depths two and three. Exclude eight warm-up anchors from command timing. The representative cost model charges each measured proposal-time change and the corresponding observed acceptance.

| Metric | Ordinary BF16 FFN | Fused BF16 FFN | Fused INT8 FFN |
| --- | ---: | ---: | ---: |
| Two proposals, wall median range | 3.61–4.32 ms | 3.59–4.29 ms | 3.12–3.69 ms |
| Two proposals, full matches | 768 | 768 | 757 |
| Two proposals, accepted teacher tokens | 840 | 840 | 844 |
| Two proposals, modeled median tok/s | 55.60 | 55.64 | 56.69 |
| Three proposals, wall median range | 3.70–4.41 ms | 3.66–4.38 ms | 3.20–3.78 ms |
| Three proposals, full matches | 768 | 768 | 754 |
| Three proposals, accepted teacher tokens | 1,027 | 1,027 | 1,031 |
| Three proposals, modeled median tok/s | 55.37 | 55.41 | 56.03 |

The fused BF16 arm matched every proposal and saved at most 0.04 ms. Retire that geometry. The fused INT8 arm saved 0.49–0.63 ms at two proposals. Its additional immutable weights and scales total 191,351,808 bytes (182.49 MiB).

Four alternating cold 4K worker pairs per mode used the same candidate build, the INT8 vocabulary head in both arms, and two proposals for seeded sampling or three for greedy. The 16 requests used context 4,352 and zero input reuse, sampled and emitted 256 tokens each, and matched the pinned output hashes. Vision and DSpark remained resident.

| Mode | BF16 FFN median | INT8 FFN median | Median paired change | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Seeded sampled | 52.07 tok/s | 51.96 tok/s | +0.04% | 2/4 |
| Greedy | 60.89 tok/s | 62.98 tok/s | +3.57% | 3/4 |

Both arms used 530 sampled rounds in aggregate. Sampled candidate rates ranged from 47.44 to 53.95 tok/s. The two workers accounted for 99.88% of observed Metal counter increments over 81 intervals. Thermal samples were nominal or fair, with 141 additional pageouts and zero swapouts. The sampled worker delivered a flat paired median at the cost of a second weight bank. Archive the FFN runtime candidate and retain the INT8 vocabulary head as the strongest repeated sampled result.

## 20261121-INKDECODE-50 · Poll the short verifier's Metal completion

Lily previously improved with a polled Metal wait. On Inkling, poll the target verifier only, using 20 or 100 microsecond sleeps, then take the original completion fence. Preserve the existing terminal-error check. Draft, prefill and commit retain their waits. Use 12 rotated comparisons per arm and width for direct and queued verification, with the INT8 vocabulary head in both queued arms. All direct logits matched serial. The queued variants matched proposals, verified F32 rows and the committed state at each checked kept width.

| Wait policy | Three-row queued paired wall change, stock 4K | Four-row queued paired wall change, stock 4K |
| --- | ---: | ---: |
| Poll every 20 microseconds | −0.05 to −0.06 ms | −0.09 to −0.11 ms |
| Poll every 100 microseconds | −0.04 to +0.04 ms | +0.04 ms |

In the first short run, the 20-microsecond policy saved 3.34–3.66 ms at the queued wall boundary. Direct GPU timestamps also fell by 2.63–2.70 ms, so the result cannot be attributed to wake latency. Replaying the short run with the same build left only 0.005–0.007 ms of queued change. Retire polling at this scope. The two stock 4K inputs and repeated short input put the effect below the round budget. [Lily host round trips](2026-11-19-lily-mtp-host-roundtrips.md#performance-and-memory)

## 20261121-INKDECODE-51 · Split the exact verifier router into native BF16 products

The short router has 128 output experts and an exact source split at 2,048 and 256 input channels. The register kernel issues four output tiles and loops over each slice in K16 blocks. First test reducing the physical cooperative tile. Metal requires a dimension of 32 for two cooperative operands and rejected the narrow K16 shapes. A register-fed BF16 M16 × N32 trial compiled, changed every inspected router word and added 3.69–3.95 ms per three-row 4K command. Retire that register-loading arm.

Next vary the number of live output columns while keeping the original physical M16 × N32 × K16 operation. Exact F32 padded widths of eight and 16 both matched, with at most 0.26 ms of paired saving at three rows. The attempt to write a cooperative destination from device BF16 operands did not compile.

For the native path, dispatch the two channel slices as independent device BF16 TensorOps products, write F32 partials, and add them in a separate router join. Compare widths 32, 16 and eight, with physical tile heights 16 and eight where available. The selected tile is M16 × N8. The M8 × N8 pipeline was unavailable. The selected shape retains the source split, per-row output indexing and existing expert selection.

| Native candidate, stock 4K | Three-row paired wall saving | Four-row paired wall saving |
| --- | ---: | ---: |
| M16 × N32 | 0.91–1.00 ms | 0.99–1.28 ms |
| M16 × N16, narrow-tile repeat | 1.16–1.37 ms | 1.13–1.53 ms |
| M8 × N16 | 1.16–1.31 ms | 1.15–1.45 ms |
| M16 × N8 | **1.18–1.45 ms** | **1.20–1.27 ms** |

Use eight rotated plain comparisons and two separate timestamped comparisons at each width. The native arms matched every checked router F32 value at all 48 visits and every checked full-target F32 readout on the short and two stock 4K fixtures. At three rows, the M16 × N8 router plus join used 1.20–1.43 ms across the stock inputs. The same-run projection controls used 2.40–3.00 ms. The candidate uses 4 KiB of F32 partials inside the existing source-verifier memory reserve.

### Paired cold 4K worker screen

Pair the vector-loaded source router and native M16 × N8 router in one sealed build. Use the same verifier options and INT8 vocabulary head in both arms. Run four alternating cold 4K requests per arm and mode, with vision and DSpark resident, context 4,352, zero reused inputs, and 256 sampled and emitted outputs. Seeded sampling uses two proposals and greedy uses three.

| Mode | Control decode median | Native router median | Median paired gain | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Seeded sampled | 51.51 tok/s | 52.72 tok/s | +2.69% | 3/4 |
| Greedy | 62.24 tok/s | 63.36 tok/s | +2.01% | 3/4 |

All 16 output hashes and paired round counts match. The sampled proposal-plus-verifier interval fell 1.00 ms per round at the paired median. The workers accounted for 99.85% of observed Metal counter increments over 81 intervals. Thermal samples were nominal or fair, with 321 pageouts and zero swapouts during the run.

Repeat the four seeded sampled pairs in reverse workload order, with candidate-first loading and reversed arm order. The control median was 51.83 tok/s and the native median was **53.28 tok/s**, with a range of 52.20–56.30 tok/s. All four pairs improved, and the median paired gain was **2.80%**. Output hashes and round counts agreed in all pairs. The combined proposal and verifier interval fell 1.11 ms per round at the paired median. The workers accounted for 99.85% of observed Metal counter increments over 41 intervals, with fair thermal readings. At the repeat's observed round counts, reaching 60 tok/s still requires recovering 2.23–4.64 ms per round across the four sampled requests.

### Parity and admission

The sealed native build reproduced the serial 96-output continuation and complete committed state at depths one, two and three on the four-token fixture and both stock 4K inputs. This covers direct two-row verification and queued verification at three and four rows. All 45 rejection and capacity-wrap controls passed. Abandoned commands reset the session, and a source-mode snapshot was rejected in ordinary mode.

With flags unset, connected short and 4K aggregates matched. The short full-model run had zero unequal pinned F32 elements. Four stock 4K fixtures matched 16 F32 vectors, 768 BF16 residual blocks, and four exact snapshot replays. The native worker reused a 2,048-token prefix across proposal depths with exact continuation, cleared reuse on cancellation and mode switches, preserved the accepted-stop token counts, and respected short and long-context admission.

Use the native router as an opt-in with `STARK_INKLING_EXPERIMENT_VERIFY_NATIVE_ROUTER=1` and the source short-verifier router selected. It runs inside the existing 96 MiB source-verifier reserve. The measured additional verifier workspace was 60,473,344 bytes on both stock 4K captures.

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
| Live transaction profiler inkling_slice | `bf60613355fd29091d0f26d35db45ce0ac5ee2c38d627318c61060faf352cdf3` |
| Live transaction profiler scored-source.patch | `188162abbf9370b5bc337018196b8195bae3087853885ad0349df7bb5e606849` |
| Live transaction profiler stock-s4096-00.json | `3e872c124a8e9292bb3a46c36f2c2c7526308dc1c1379396bf60d320eb5a5` |
| Live transaction profiler stock-g4096-01.json | `167c0b994dd46f2726bbbb900caaa6810f685a2916c4449e3ac4469b541a0158` |
| Global K/V screen inkling_slice | `1b91ffe054035ce5fbb606ad6a968fb3c18f6acf86a5a3f41b4e7cec78d14b07` |
| Global K/V screen scored-source.patch | `5713d34e9f9c7a9ec1070bd07efebbc329983bf153ed99c95a4161af02637677` |
| Global K/V screen short.json | `d7543b6c7fa3a12e696bbfd2e4148c0a1778b6ae87a74b5f5041e804cad9baca` |
| Global K/V screen stock-s4096-00.json | `10985491a8decd617e932fd5052c82602b76c5c67835782e80afc3d70a0c039b` |
| Global K/V screen stock-g4096-01.json | `b2327e2e60850076400eb4676f4ebe3fed88d976c2efe52394aa96ffd5c12e70` |
| Global K/V worker stark-generate | `b0f7a9b3396fa26011b85f30811cd340b68603f0872037fa87d6b731fe3e465c` |
| Global K/V worker worker_probe.py | `3640dccb66b03e35de3bc9a7dff1d19c8eaa73cd17c603e556a6c660da0c747d` |
| Global K/V worker worker-pairs.jsonl | `7b03d3c87b75f2834aa53964c9ce246141d27d0a1be3ab2df8c205bf2f829385` |
| Short verifier router inkling_slice | `567bd213c5c4f4b4483251eb5d046eddeed8218913786ee7648c40ce66bca8cf` |
| Short verifier router scored-source.patch | `21c8b894166b99b626096394ba3a9177d0360fd189b8050167db80241cbff112` |
| Short verifier router short.json | `d4b099174d86c24b82293edf7dd10bc5c8e1501733dfde27fa79be7034cafeb4` |
| Short verifier router stock-s4096-00.json | `c5b7241c414b2e108875261115f0c5a98aa9be66af025aeb7a14931b0c55bf71` |
| Short verifier router stock-g4096-01.json | `ff6588f52f372aa454a9f44ec205157bc497608013acf082d64b0e3ce0462752` |
| Short router flags-unset parity final/flags-unset/slice-4.json | `3ec9bea159f2df8ab1c508329cd2b102cb53e672c740ae3fbddc9ba677cbfa63` |
| Short router flags-unset parity final/flags-unset/slice-4096.json | `c4a5cefe72c1633b5a56a7092e2e47479698d79936933e66910289b867c40652` |
| Short router flags-unset parity final/flags-unset/short-both.json | `9857fe852764ad2785a875b523921b4dc81779138460ff38b6e9e68c7aa17823` |
| Short router flags-unset parity final/flags-unset/full-0.json | `97d8823b984369ec3a5bccb2e0f55f9e0c5dd30c85dd991b6aacad56a5595f87` |
| Short router flags-unset parity final/flags-unset/full-1.json | `c37b8fe4ea4e9f9614ea8552c9702480e7425e08d2867fad98b3ef77f53d946e` |
| Short router flags-unset parity final/flags-unset/full-2.json | `78465962571720c6cadfafbe1de2c4f8f7461244686d95485cc397baa885f6d5` |
| Short router flags-unset parity final/flags-unset/full-3.json | `98a31d0a1c9eff82f6278b8d158e642b5fba19b90614aa280bd68296faa25748` |
| Short verifier router worker stark-generate | `d883d29a92cf06d639860eba23bd640abc1cc0fae95d87f5055f5d16e6f7e2ec` |
| Short verifier router worker paired_worker_probe.py | `89abb70b0dbaff5dacd67d5e109d42ef399e620bad9c9024d7443b89b96355f0` |
| Short verifier router worker worker-pairs.jsonl | `0cf205b37ff7280384e5c8e90967aed87e7a974710b225296e5b0191ab79a539` |
| Short verifier router worker worker-host.jsonl | `2706ab072f958de170c8ab9d1ae991b1c3e34e8eaea14aa593a5e7f0fb87058c` |
| Short verifier router worker flags-unset-readiness-first.log | `b754b166fe5a3e574979bf67197e3de4cc8be5b86313eed1bde26bf4b2a2a727` |
| Short router flags-unset worker_probe.py | `c4bfb8b9d96381457015832762774cc93d3cf2872a3b5b685308baed90e1c54c` |
| Short router flags-unset flags-unset-worker.jsonl | `bfdde50e87683c2b6bb3ac997cca255790af6cebc89484a98be07b6333c0aace` |
| Short router flags-unset flags-unset-stock-profile.json | `8d79302c6c127216a3bafc62ec48c941b12391af93dffa617a1961eafa09c2a3` |
| Wider NVFP4 output tile inkling_slice | `d7aabafec1f49e0408dea17decf446462f7f1f0aac1ce140ff4bdbcc434cb60f` |
| Wider NVFP4 output tile scored-source.patch | `753da349f7d5c0e89fc0101479f314247032cf0ce6483d7cc3108b98459a0409` |
| Wider NVFP4 output tile short.json | `abf90d2d49638d6541a4bd69d72558356e0f6f470be7f6d895f804ad7d6d02e9` |
| Wider NVFP4 output tile stock-s4096-00.json | `a5805e0840c9fcde1f171b2411e6bd83d1b885e57279ea6b764a764d4a9be846` |
| Wider NVFP4 output tile stock-g4096-01.json | `18e81700512ba5a6f48ee029869b351ee4b91212146d0e52bed00f7f46861c0c` |
| Sampled depth bound inkling_slice | `97e6a4a3ba10b82e0a4b721fdeec67b32670232b438f781ff509817f32b87aa5` |
| Sampled depth bound scored-source.patch | `ec9130f96890d7cf0e7e543c55f8ef4eae6d23e8bbc68fc0d2317c0bfb6fe09d` |
| Sampled depth bound run.py | `5690e81545c13d38bcedfa6da9d639b1a6f164cd022b0923cd5007f4e602b285` |
| Sampled depth bound result.json | `eead7dc77993ad7e3625fbe5d1a945859e16ab2d86ec9c3ce854f471964509a4` |
| Sampled depth bound pre-draw/inkling_slice | `63b9f03a7d846ba4a391bf3f0dcd4f5f26bae778cb9db4bf4e11c78cacd067bf` |
| Sampled depth bound pre-draw/scored-source.patch | `73b8668b56ed618c4a50e76f3fffdbea796c30fb59f6f03de68df6118d0af6eb` |
| Sampled depth bound pre-draw/result.json | `aec3adb0e955e0c045fa97df5f7773fdd8e3b58b92171f5c44cd63691858c063` |
| NVFP4 read and decoder screen inkling_slice | `82999a6511194c32cf0cea39c6b7445e66e422d71853c5756419a76d477ebf13` |
| NVFP4 read and decoder screen scored-source.patch | `7154ff6c3319203512f6486aaa250e59e65abf28431e4bbd6d43041a7edb400e` |
| NVFP4 read and decoder screen short.json | `669f7e7421a3b536fa8071deaa3a4cc5ae076f76c1251fc59064b0c43c897142` |
| NVFP4 read and decoder screen stock-s4096-00.json | `9523542b0cd2b5d00d61473a799d58daba73cf1b6bbe3b6f98007d891155c443` |
| NVFP4 read and decoder screen stock-g4096-01.json | `9a7239e50f69fb59a9840757c21ac1be24ca167b1cb66c6ef4ec463b8118c830` |
| NVFP4 read and decoder screen check_coefficients.py | `ed78be1b1323bb1623bdded6a146d7a736b8a75488160f01c9cedb96efae79d2` |
| NVFP4 read and decoder screen check_coefficients.json | `0bac80e70e445409764e98a7ad477592f07cc221445f950dfd7bf6c8cd83ac9b` |
| NVFP4 FP16 element product screen inkling_slice | `f8d9d57856bc16ce2f975ef21f809aae2a886f4e42318afd8f525cc24d59b8fc` |
| NVFP4 FP16 element product screen scored-source.patch | `55e067aaed16489c9dba24f40196edce7a436067d958e33e94c7e4e619a7bb7f` |
| NVFP4 FP16 element product screen short.json | `77556d893227adc28eb0f302be74683d34d694afd574d3628f1224a828ed41c5` |
| NVFP4 FP16 element product screen stock-s4096-00.json | `41b181eb33dde18b20c47dbcf2553c39a516705f6e86dedb77866317ddf1be83` |
| NVFP4 FP16 element product screen stock-g4096-01.json | `4c056566fd6701d9cce42e8dccb5ae09e36971dd4222738e93d031275ef399c2` |
| NVFP4 FP16 element product screen check_products.py | `3a7b873ad84c89af15e5079bde739282cb26d922b902f6c0ae5f1623a69a9ab3` |
| NVFP4 FP16 element product screen check_products.json | `1b6c103afed575b732a696aa330a320e7651ab4cc86fcef63908beb05069057b` |
| Affine FP4 retained build inkling_slice | `86882f7bf35703102be8c88789f2424bdfd71cb7b1d211231b1e77bf2d0ba66e` |
| Affine FP4 retained build scored-source.patch | `cc9534b022d31844717803f277a756946e151d479ac68feb098747f3ec12cd2f` |
| Affine FP4 retained build short.json | `41ced58ae70b776cd0b01535daaf841d08a2aa3686029d3945c313d04c390d72` |
| Affine FP4 retained build stock-s4096-00.json | `47d5c913533593d0774beeb4673170c6f83b3297e0242e53901a78a51c885d4e` |
| Affine FP4 retained build stock-g4096-01.json | `254bc6adf3d063714e5cca4ff43b9ccb61d2a4e6298e5a524eee833caa82c8eb` |
| Affine FP4 retained build flags-unset/slice-4.json | `3ec9bea159f2df8ab1c508329cd2b102cb53e672c740ae3fbddc9ba677cbfa63` |
| Affine FP4 retained build flags-unset/slice-4096.json | `c4a5cefe72c1633b5a56a7092e2e47479698d79936933e66910289b867c40652` |
| Affine FP4 retained build flags-unset/short-both.json | `9357b0c5d286f9272f47c80bcfd0eea083022ef8542e127c698ac9085b7861e5` |
| Affine FP4 retained build flags-unset/full-0.json | `8e5390236d3da3f73c50e2da3c94b8711f5263a4013a128759693112281c112f` |
| Affine FP4 retained build flags-unset/full-1.json | `a75bbafce59c724c433c8354c19d979e87f29f2d1c7b1a9d885775208ab1f206` |
| Affine FP4 retained build flags-unset/full-2.json | `f67cfa2b8dd4a29d1898ef2bcb23fe9d0be71933ca0d60fcaadc1e73fb867b96` |
| Affine FP4 retained build flags-unset/full-3.json | `2ae7883d68df5fa710431bb486a2d7fafa6c44a401a1f69caf540767d0bb8711` |
| Affine FP4 worker stark-generate | `99cabd382cc554288145bfcb70150f6e080da78d1e72d4edc4ee5d3d1c59560b` |
| Affine FP4 worker worker_probe.py | `e0ef33f19b12e4b73a92306565692610f1ca0cfa3d15ef189681f9295a684cb8` |
| Affine FP4 worker worker-pairs.jsonl | `7183447837802ee8f2edda725572cd05ce34e8e0141fcac52eb20afeb0f75eb8` |
| Affine FP4 worker worker-host.jsonl | `8b7a5fd5ddf439c2f93b03e96d252e0dd5a58cf4c5538638250ddae7bb426ee2` |
| DSpark proposal stage profile inkling_slice | `b1de505c1ae6a6911aab2111c0d7ed64ed3aba26b5b01cb6a4804fe8d68fcb0d` |
| DSpark proposal stage profile scored-source.patch | `e5c605c710ed2ccd157b9982dcb956765d5f9dbef7a4925caffbe11a5631ad44` |
| DSpark proposal stage profile stock-s4096-00.json | `753bd5066802dd2620cd5ee851b47250aa81b0ebb8e3edbf6a67d80b3703ead7` |
| DSpark proposal stage profile stock-g4096-01.json | `6594f05ae38d5d30f79b2b306645600a78d435e8fb923a104499683233e85314` |
| DSpark vocabulary tile inkling_slice | `3b96ebb50a2f44f19f7c6e344ba068fe40a653ddf16c4ccf5c9f96a72aafd479` |
| DSpark vocabulary tile scored-source.patch | `e4a751452332816b550b084bf3e525152c0a05cf6e1238c445597dd481845cc9` |
| DSpark vocabulary tile stock-g4096-01.json | `4a1b79531b4b8558d1485e73e75dd4aa4c74e2cdd29aa94c6be7527f57f716e2` |
| DSpark vocabulary tile stock-s4096-00.json | `83cdba7bc57853a4107ad5d148ddd19eb020a4f1a5d4ea7dc69518d52e1bd1c1` |
| Overlapped verifier inkling_slice | `9c5e7e761c926e04f4cc09b1e9ed1fd37eed89df358bde5a54d0be0ad7b5d9ee` |
| Overlapped verifier scored-source.patch | `8456d7c39565fda903b408d5f99f2ae628d347418ca84bfee9ddafe868199001` |
| Overlapped verifier short.json | `0de6c5263801e0b2bac726665e16627c7a1b07c003c75d495cc8725ecb6a127c` |
| Overlapped verifier stock-s4096-00.json | `d3de1b45715179104bd62a8e9cecd9838c9aad2d4923cda3ac8d95c98ef8bf46` |
| Overlapped verifier stock-g4096-01.json | `e943dc45dc6ead0a4e4e03e0c4ded551ab154695b8bf0bf2693f39e9faf4d160` |
| Overlapped verifier flags-unset final/flags-unset/slice-4.json | `3ec9bea159f2df8ab1c508329cd2b102cb53e672c740ae3fbddc9ba677cbfa63` |
| Overlapped verifier flags-unset final/flags-unset/slice-4096.json | `c4a5cefe72c1633b5a56a7092e2e47479698d79936933e66910289b867c40652` |
| Overlapped verifier flags-unset final/flags-unset/short-both.json | `58eddd281b2bd8ea1d502bcdd764bfbb3a93a5c3b165da8a110e2cbb4f45d55a` |
| Overlapped verifier flags-unset final/flags-unset/full-0.json | `ab63da118a07cd53c3b5f72caf6b3d3dca053cda3420daa9eaae80a8fb44ecb9` |
| Overlapped verifier flags-unset final/flags-unset/full-1.json | `d7e7b69a82d2163116b092dd5a687a06e61f802a8e508d00b4b95255e89c057f` |
| Overlapped verifier flags-unset final/flags-unset/full-2.json | `ef5633dc607d2293ee3017cd2a12c7aa2a8943c140132d5a07b03a00aa1ffefa` |
| Overlapped verifier flags-unset final/flags-unset/full-3.json | `336af745c75edbec827071aedaef4adce1aa99faa9b30bd0f554c4acdcb9f9e9` |
| Overlapped verifier worker worker-scored.patch | `8456d7c39565fda903b408d5f99f2ae628d347418ca84bfee9ddafe868199001` |
| Overlapped verifier worker stark-generate | `03b64caa5cd91a3db693122ddf13917ccd28b81aacbdec104171ed0a2537a4eb` |
| Overlapped verifier worker worker_probe.py | `050f746cd3473fd82c1ec96b5741e3f21168e46adc706cabe066d1c4bae3bd1a` |
| Overlapped verifier worker worker-pairs.jsonl | `23df6697e417b41e57f108cc4467a8cfc914da1c17102813f5012921d3945f31` |
| Overlapped verifier worker worker-host.jsonl | `254e8cd636528e75df6e69360e9af622853fe4cdbdf92b629b23e20d83fd82fc` |
| Overlapped verifier repeat confirm-source.patch | `f68590d1ee99e8c3ce295853857d18601aa763872a899e28fd1095b03aab1c23` |
| Overlapped verifier repeat stark-generate-with-telemetry | `ada452a22f7959c7fa88b1005742a8e2fe6b329612a68e502c08b22f3b3647bc` |
| Overlapped verifier repeat confirm.py | `4ce02fc558929455443871b2a791e8c00d0c85cb452866e967f561a7e128b1a9` |
| Overlapped verifier repeat confirm-pairs.jsonl | `4af369ca34f4cbb4094172928c8f11d38ef6536006777f628022af14fec9a462` |
| Overlapped verifier repeat confirm-host.jsonl | `0e1c1cdbb81e52a6902d6c4b798b1bc900c4f0d0e8f0929c9536bf9e724ff3e1` |
| Overlapped verifier lifecycle lifecycle.py | `58fd4621e615d0531d92533268c405dba777e4eeb447e84503fb7f4ea62444dd` |
| Overlapped verifier lifecycle lifecycle.jsonl | `d1b51cd09652bd043eb12d6546e89535af29ed286b5f587400a88200975afc6b` |
| Single-buffer verifier inkling_slice | `4aa240edd472e2154433f784f4e46c38c74deed633c46262fa425b787280e8cf` |
| Single-buffer verifier scored-source.patch | `4f33f199c74a359fa22dc277ba66fd7a8aa82a2f80448f79d6bbe062a3b01d79` |
| Single-buffer verifier short.json | `80b01c003cb23fef83a8f2e775777a97df26a23921f414d6b276006640baac21` |
| Single-buffer verifier stock-s4096-00.json | `3dc2791ca904d6969c5f38d4363d2ef924cf5518e6fe90f2a2014488f4d1c043` |
| Single-buffer verifier stock-g4096-01.json | `9cb07c8852a2507a40fe87b6a50a9850f9b45535b5908e772aa0f3662e7ae7c1` |
| Live-width dense verifier inkling_slice | `44bdd778c06e8448e8de9b0bff45914be11dae5da891d3ea882d8430517ef0e6` |
| Live-width dense verifier scored-source.patch | `36c60470d4324dce323da89a93d4dfb6a70f67cf3556453f397481a4b547afed` |
| Live-width dense verifier short.json | `b9ab603c6c1d4e9a8cdbecc341e87ce11ae532bc1ede095baa460be80ad4f253` |
| Live-width dense verifier stock-s4096-00.json | `81ca185fcb3fd552d986d5ee8693964bb8571cec77d0d049a57ec37d6aee3205` |
| Live-width dense verifier stock-g4096-01.json | `585087db4e2d1ab79d3c855db9f180d39907e10da77cb1a98da2e7ef57af69af` |
| NVFP4 tensor budget inkling_slice | `eec502b991f4628bbab137c5749f7c23ad7ab06e3b73e7f95b0eada24e4a8277` |
| NVFP4 tensor budget scored-source.patch | `f2de30478ff0d07ac2ffb28f3e0a0213507c211173bcf76d4ec9ecbb33c761a9` |
| NVFP4 tensor budget stock-s4096-00.json | `57e87288d1f76c39e34edc55db27a7b4baa6f28bf901ebb5853b696a377bc515` |
| NVFP4 tensor budget stock-g4096-01.json | `32a4e840b539f3778af20b86349f67cc24a83cbaba7a455e37e24261e4c2d994` |
| NVFP4 K16 accumulation screen inkling_slice | `d7634e460bb27bff7ad400af247f65dc4838dc2b061186abc6ab7ad40bc1041b` |
| NVFP4 K16 accumulation screen scored-source.patch | `3466e3a2ecfdeef2a3f13ea26b7e15719ebe2b30f1a967ba250b340c595707c9` |
| NVFP4 K16 accumulation screen short.json | `a7ea69536cd9f6c6c13e66dbd7334a6bb08c7cdb16abb71f221c5e5383290775` |
| NVFP4 K16 accumulation screen stock-s4096-00.json | `b9e1f0b8badb7204df4dc9793d2a5de647fb5151eb2b340d64f6aa7bd69f2906` |
| NVFP4 K16 accumulation screen stock-g4096-01.json | `b73c92b4a08f128ec858d2f307bedab84d9abaf80ffb27045267e6776ea93f8a` |
| NVFP4 register tensor budget inkling_slice | `5a56a8feea6441b901dbde90ade226758c7c07d1a6d99e80be493fda9ef8de08` |
| NVFP4 register tensor budget scored-source.patch | `a9f4f02772551339345ab86ce1243590bf32997f647c0e59cf73fbec25878d4d` |
| NVFP4 register tensor budget stock-s4096-00.json | `625317a00dd960a773c2d199c2c5a270b72cad91186442e7d7c111dc3b125c16` |
| NVFP4 register tensor budget stock-g4096-01.json | `fce4a352ffa3b6959c2fe9916c7aec588378955a1913d9a814e9038352aa2ec4` |
| Ranked branch replay inkling_slice | `2e6f29d2767a8e362e5056c13aa411afb031af23abd436a6397b46ee476af801` |
| Ranked branch replay scored-source.patch | `385dc0c0bde01690f7a49fd30b5d4812bd2b8f21ce1a82d2233b0719ebc80731` |
| Ranked branch replay run.py | `bd7ba11f10e4237263716656a523e6a27ebefac7cce55bbb7bb571b6a5a6b006` |
| Ranked branch replay result.json | `c2253af4b65ebdf9e0da55ced506bc0a083e2a607278f65a2e141ebcc64cbbc8` |
| Quantized draft replay inkling_slice | `5fd46685d37336f0b2e85eed2e4bd6b05b2d1c396dab570b4130ebd84e0cf60c` |
| Quantized draft replay scored-source.patch | `7ed11662dc64e081cad2005ea1daa6876f45c2a2b879a33b61de670bcbbc5a06` |
| Quantized draft replay run.py | `db424caa265a15f61094c6db453002fe8a4cade7a5ba587f73413f2359e88afd` |
| Quantized draft replay result.json | `2c140b8edd79b811825bcafac036663f82581fc611985037575d50d9709e0368` |
| INT8 initial worker stark-generate | `5758fc0430b8733564c61027bbcabcf6100b1bf1aa8dc6a9a76005db388de03e` |
| INT8 initial worker source | `1b581d149e18f1be19e43a86713bf166e138fc98c7572ca16b3a1b52d9a5009c` |
| INT8 initial worker pairs | `0a719402158f4edccf234973ec2ca319c6acf8ecc226022911fa16dc59c22997` |
| INT8 admitted worker stark-generate | `ee05d63c6aa2ed1ae40e1a8f6e7d7e1120db2cbce5ba85401659503bc2593079` |
| INT8 admitted diagnostic inkling_slice | `3e6e2c4192af1d6ab52b2ba9f3bfc90043d77ae8370e0f7a41ad9caf3ff66082` |
| INT8 admitted worker source | `60ff5c82979a4e58eecf7bb851c071163c57b24becb4037e11deeb494c9b5ec7` |
| INT8 admitted worker pair harness | `7158ac77cb1b818e79c41a3a67b9e178f73da867758e48188749e1873d3589ce` |
| INT8 admitted worker pairs | `287d747833fd5e5470aea84cb422856eb552f8360a52b121662413e5dc09d1f5` |
| INT8 admitted worker host | `269c612d444f172dfae166f2b7c2493a5d81b08f9a4610674ce71ad884b8142a` |
| INT8 branch flags-unset parity aggregate | `b8a5b540c43b3942a458316f8d3f2943ed64beadfbb40fe803a93111aac3d011` |
| INT8 branch flags-unset parity checker | `1b7ca3b5754cee623336984f3d3a1c761cdf485a2e0e549a1fe52a211da8e9cf` |
| Exact integer K16 Metal source | `4bb5affeee76c633a60bc7a716c6aed17b6937ea16ac445d31435ef9975d6668` |
| Exact integer K16 Metal object | `a5351d48ebcd079be0b7de323170f912984d1d4cf6d224dc37701e16824a220d` |
| Exact integer K16 scalar script | `08c28a408e1b3cbd11fc95ad75c23d88f276d4ac27c66eb49b8629ca62b341b7` |
| Exact integer K16 scalar receipt | `cd68a8601d29eb49a3b2c219bf290b102c730583e45275d5b880052d99825cc3` |
| INT8 native parity short | `6d808c16b9833e1c402eb93e9e7eebbc66d4ea0242704c6d71c11b275d4b8a78` |
| INT8 native parity stock-s4096-00 | `9c3a151ff93d85479417260857f7b058d280a36ca1b0a9481287a8701db42836` |
| INT8 native parity stock-g4096-01 | `0d0886f26f4173a6a6d0f80c3725a13886432374c27b988651098d35baf7d0a1` |
| INT8 branch flags-unset worker windows | `3bd9387036acd282faac61005d5d817349e1163f1117aa381f96c5c9b052e0aa` |
| INT8 admitted lifecycle harness | `2f99879ab363e99073cea504a3af0b087b6b8a4e72ce1cc3b7347a8662f9d3ef` |
| INT8 admitted lifecycle receipt | `d1b51cd09652bd043eb12d6546e89535af29ed286b5f587400a88200975afc6b` |
| INT8 admitted idle memory harness | `7f811388682f744221ba02a8b133b4185ef62a26bfd7e62a908169edae55eefa` |
| INT8 admitted idle memory receipt | `7be7d15dc9168144bce33ff38f6dfa224d16d89bb7550715346515db9b92109c` |
| Live integer experts inkling_slice | `2a9a0abc93898c23b7fa97ab2996fa14cf426ec028ea2dfe282486d2d30f9322` |
| Live integer experts scored-source.patch | `411015de1a9b2044da9f9c068b95838859481e1d128f98afac519e740d07f94d` |
| Live integer experts score.sh | `51995381bbd22eaa3817839b76f4a99f1f20f698ed1bc57357d8a7672959179a` |
| Live integer experts short.json | `1dbbfc62a2f29d0f24f4ba71e954afc5cff6b244e4e9b00b5e371b4be0679bfb` |
| Live integer experts stock-s4096-00.json | `2779065be355e3254b3c9a42a53b6d32a066e46f8a53fe51d2a5b4c3d15861d1` |
| Live integer experts stock-g4096-01.json | `65f14f684a1bfe810503ddb1e6bcd72ae1d86b3cb411f2748b47e38ff84b008d` |
| Fused INT8 draft FFN replay inkling_slice | `6fbffe09cafcffe8c1e0db88b888a53335b591296f04253fac7360fefd1ff5c0` |
| Fused INT8 draft FFN replay scored-source.patch | `4e948bd50ed10834fdb8a6208e1488502a2ff2ab12ae88caa4ece727b4105866` |
| Fused INT8 draft FFN replay run.py | `16ec7f0d9b93e25fb4a3a16930866416cb5de47e61e4f8405ad105e140fcd3e2` |
| Fused INT8 draft FFN replay result.json | `9fb0ac82db6764fcf53b2cd5a5055d3393f82f46acffe2568e5cde2e82ce6fe9` |
| Fused INT8 draft FFN worker stark-generate-with-telemetry | `734b239dc6941066a36534e013b6089505abad18a62c9b2cf36241373516a499` |
| Fused INT8 draft FFN worker worker-source.patch | `d68b8e8276e796c50b02517d8c02c01ad6414d02a172d975d2d73fccab5c1408` |
| Fused INT8 draft FFN worker confirm.py | `929ca6485760d6d21d844a4b7e4c80a844b0676266b2298bf864ba537a522274` |
| Fused INT8 draft FFN worker confirm-pairs.jsonl | `176fb01828e2a73ea949fa218cabb8cdd3eeed4c8c6bc31f4a8df20a4f3080b3` |
| Fused INT8 draft FFN worker confirm-host.jsonl | `7d80848ce8566c27f3bba85db000a191601c6b6f903fe14f4bf3518c2e44d427` |
| Short verifier Metal wait inkling_slice | `2bafaceb3601435b73297769dcb5e5f1485e5564b528efdafd7cf5c061030810` |
| Short verifier Metal wait scored-source.patch | `ae04b3ad8f59071a497ffc4dba3d3fe89bd2045be41adeac747fc5993475b435` |
| Short verifier Metal wait score.sh | `96433cc0091803ab3308f465b13a99562b21630ce18a2382615061127a263049` |
| Short verifier Metal wait short.json | `ba355f5f071246b719217eecc25d84bfbaceb84db1362327a4120c64b4bdc703` |
| Short verifier Metal wait short-repeat.json | `4f83351c18bedba9cebfaa2570a4a3a1f151604970aeebb0399569f327918177` |
| Short verifier Metal wait stock-s4096-00.json | `252bca6dcd2ff07e82649751c518280273d4a8e0aadcba160761f28efbf5f086` |
| Short verifier Metal wait stock-g4096-01.json | `5485cd87e6ef5b004c01d29e66189c74313718d3fc313990e357caa2f707b639` |
| Native router cooperative inkling_slice | `2847557156dd1d93bd80edf3b9983624b7f548d20e6b870017b2689f0cc51f1d` |
| Native router cooperative scored-source.patch | `3773bb2039d17f45c21d349c55c63076fa09c59f8a86f08353cdbd402a7e806d` |
| Native router cooperative short.json | `77eb3330fd6b99da8cb895fffae77621fc8e132493ebe8595f69b36811f7305a` |
| Native router cooperative stock-s4096-00.json | `e355d766578a90cd75d5b8e0f5a2c1a8a70e3ab606a6b84f57abf05fd5ad62e6` |
| Native router cooperative stock-g4096-01.json | `70c04b5991f183282f88973d1f35e35786dac3a1da9902d4bae5e0a8eb74d717` |
| Native router padded inkling_slice | `35cd7835fcb45ce83276679ad5a2a3ebcf0eba1f28188052a3f6d43f282b8855` |
| Native router padded scored-source.patch | `fe0827d4c833e48a336c0877371894036cedd55c650b64ceb5ccf79cf423ba40` |
| Native router padded short.json | `92c83a5ce7642aaac3acd986d87d9b0637ff8921221a0d81588dacc7adc77486` |
| Native router padded stock-s4096-00.json | `1d2151d035b521e6dfef551fdc5ec6ef4c68531aa7ed2b2906f2e9084fccde4a` |
| Native router padded stock-g4096-01.json | `fed359654c411d440e31b2b41ff5330fab7bff9e2828e20014dc80e2ebaf8249` |
| Native router split device inkling_slice | `303bf88a398e8677dc5d2eef021c4e1e5ef6743b75be248148dc59b7943cf72b` |
| Native router split device scored-source.patch | `da626da7da3d38fd42ec7303105578349b2421176be58e33aecaf6cac2840f14` |
| Native router split device short.json | `3ac51ee5a325a6da6c23520a51dddf57930ccd8496ecd1f9d0c167d8d382f764` |
| Native router split device stock-s4096-00.json | `aa06ae7c620f1291b74f6f6eafe9804f160abeef90052925e3182115f7979141` |
| Native router split device stock-g4096-01.json | `68b72838d50146fd8615e08ec1132045f3c6600043cd03086149233b500d9f1e` |
| Native router narrow device inkling_slice | `5cb8e303cd949bc438e7f61d074c46fe48c955bda6ad32bb7f4c7567da73eb29` |
| Native router narrow device scored-source.patch | `8a6449075f5d4d9d36d23826f90e4e116f0d245abd18c555a2c1122123776cb3` |
| Native router narrow device short.json | `9579dc3ee7c9b7ed1464276fb0406827881a8d5eb86c89211b2d076d1e15767b` |
| Native router narrow device stock-s4096-00.json | `bbe59be3b642ed37f5e58d18a02ba75300d381d44aa83bd33d61ee41ccbbb7fe` |
| Native router narrow device stock-g4096-01.json | `200bc8e26c3c8e3721b9f522e71d4fbea4c225f116caf80aba2eda43085051dc` |
| Native router worker stark-generate-with-telemetry | `7c8a23b27d174c92b2e467c408d718a32c1aa7d34801a2bef03078b09ce4bb2f` |
| Native router worker sealed-slice | `f305cad83bbe915e2d8a29db9bcea9a810367fe87994a57f66043a04b059e118` |
| Native router worker worker-source.patch | `286ee27da8da95721b075dab0d163788084f15db8a5fa8d6283fabefae2bcc69` |
| Native router worker confirm.py | `9e80cffda30f57e607a322e7589f935f5fea290a52f2d1c57fbd5876cc64712c` |
| Native router worker confirm-pairs.jsonl | `59155848c55a862cef1446951d6ba67a2d84c178bc699372d5c289e302a44f85` |
| Native router worker confirm-host.jsonl | `bc1b803718d6f531ee212f53dbe823a887519f51d8a48c988a6fc889a26b7648` |
| Native router flags-unset parity | `b8a5b540c43b3942a458316f8d3f2943ed64beadfbb40fe803a93111aac3d011` |
| Native router native short proof | `06fd93651065cd0a0cc954f54213b0873164e7155233950e7d7a846f730234e2` |
| Native router native stock-s4096-00 proof | `8785eeaa08d47541b22f200c36b7551e715bf54b1de5dc3381dbf655587bc775` |
| Native router native stock-g4096-01 proof | `c45febb22e0b045b86d3f3937b3ffecec2b267f5618a72a109ba0166b6e703b2` |
| Native router lifecycle harness | `c49423bda00cd62d8154f4b48f46be8912bb0553e8f0e6720b484515e4f223b5` |
| Native router lifecycle receipt | `d1b51cd09652bd043eb12d6546e89535af29ed286b5f587400a88200975afc6b` |
| Native router sampled repeat confirm.py | `d77931dfd29fe361ef040744d8cd939fe0bb4ba2f5908caa4c8a02de006903ce` |
| Native router sampled repeat confirm-pairs.jsonl | `f92d53d838089537b0f729e0d3c747ba5f449185efb9adc832e547c61325272f` |
| Native router sampled repeat confirm-host.jsonl | `0fb5780aaf2232bad8185a439a80722b82f4780afa8a6317040dc0155afdfc90` |
