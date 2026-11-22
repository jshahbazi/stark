# 20261121-INKDECODE-01–03 · Inkling 2 Mini exact decode toward 60 tok/s

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

## Decision and next target

Keep the vector singleton opt-in while building the faster verifier. Useful worker throughput remains below 60 tok/s. A promising next test replaces the small grouped tensor projection with the exact vector block recurrence and shares the decoded weights across live routes to the same expert. Time the route-selection stage separately on the same cache. An exact SIMD top-eight selection can remove the serial scan while preserving expert tie order and normalization. Require bit-exact logits and committed state at each measured width before integrating reject-and-commit behavior. Draft acceptance and verifier cost need a new combined source-mode test.

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
