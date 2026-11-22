# 20261121-INKDECODE-01–14 · Inkling 2 Mini exact decode toward 60 tok/s

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

## Decision and next target

Keep the vector singleton opt-in while building the faster verifier. Useful worker throughput remains below 60 tok/s. The short verifier shares NVFP4 weights and parallelizes expert selection while keeping exact logits and fully committed state. Source-mode draft acceptance confirms an adaptive path can matter on some inputs. The verified transaction now restores rejected state and commits accepted draft context. Reduced target cost is still needed across 4K requests. The regular token remains the key path on low-acceptance prompts.

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
