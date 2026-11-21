# Inkling 2 Mini native decode slice

Codex ran these checks on 2026-11-21 from base `0ce85ac054c810b7357fde249069255d02365280`. The diagnostic runs the publisher’s original three-pass target shard. The deployed Stark catalog and generation worker have no Inkling admission in this change. [Port plan](../INKLING_2_MINI_PLAN.md) · [Experiment ledger](README.md)

## Identity and disposition

| Attempt | Hypothesis and check | Result and disposition |
| --- | --- | --- |
| 20261121-INKLING-01 | Connect both attention variants, their KV and convolution histories, four cross-pass controllers, block residuals and packed GPU experts. Compare adjacent pass-two visits directly with the original-weight equations on four positions at 4K and four teacher-forced short positions. | The native graph ran. Both connected parity checks missed the numerical gate. Keep the graph diagnostic-only while investigating the producer of the first expert input. |
| 20261121-INKLING-02 | Feed the native global visit the reference partial, and the native expert path the reference BF16 expert input. Repeat both fixtures with an eight-row MLX dense producer. | All four independent attention/ingress controls and expert replays passed. Both alternate-producer connected controls failed. Preserve packed expert arithmetic for this geometry while comparing upstream BF16 boundaries. |
| 20261121-INKLING-03 | Encode all 44 original NVFP4 visits in one serial compute encoder at a populated 4K state. Compare slice wall time with the entire source worker’s time budget. | 27.35 ms median across three process medians, one submission and one wait. The remaining budget is 15.18 ms greedy and 16.56 ms sampled at the prior full-worker floors. Diagnostic timing passes the parts-budget screen. |

The benchmark and aggregate receipt bind each fixture digest, source file hash, locked build, exact checkpoint revision and artifact hash to the run. The reference fixture generator uses the original activation shader extracted at a checked source digest. The independently implemented native shader uses Metal 4.0 safe math. No prompt, token IDs, hidden values or outputs are published. [M5 Max benchmark and reproduction](../benchmarks/2026-11-21-inkling-mini-native-slice.md) · [Aggregate receipt](../benchmarks/data/inkling-mini-native-slice-2026-11-21.json)

## Controls, attempts and failures

- Before the native parity run, set per-row relative L2 and max absolute error divided by the greatest reference magnitude to `2e-5` for F32 messages, `0.008` for BF16 state, attention and input, and `0.015` for connected experts. Route sets must match exactly. The isolated expert replays check exact reconstructed quantized inputs and apply a `0.006` limit to the weighted midpoint and final BF16 output. Passing isolated outputs and both reconstructed quantized inputs turned out to match exactly.
- The first connected run failed at layer 10 expert output despite passing the local attention, cache write and cross-pass gates. The final build reproduced the failure on the source-shaped 4K fixture and the short teacher-forced trace. Swapping only the dense producer in the reference produced the same qualitative failure. This keeps the tight connected gate in place.
- A standalone Metal compile check initially tried the expert fragment without the shared header. The combined library compiled and ran successfully. A Metal compiler warning about threadgroup publication in the activation row-maximum reduction prompted using the already-reduced maxima in each SIMD group. The final build repeated all parity controls and three timing processes after that change. The earlier 27.43 ms smoke output and 27.34–27.39 ms pre-change process medians are excluded from the final receipt.
- Time eight warmups and 32 singleton steps per fresh process, at positions 4,104–4,135. Only one experiment used the GPU at a time. Collect host thermal, memory, pageout/swapout and observed Metal counter totals. The timing study uses synthetic ingress and cache history with real weights. It covers only the partial graph and excludes load, capture, upload of each diagnostic ingress and any sampler or output processing.

## Generation qualification

The diagnostic establishes per-logical-visit mutable KV and three-step histories for four convolutions. The remainder of the 24 physical layer graph, query-tiled prefill path and generation state owner must exist before this state can be snapshotted at complete prefill boundaries and exercised through public prefix caching. Current local ring writes and global append positions are bounded to the fixture controls. Snapshot/restore, cache recovery and cancellation were not run.

The source vision tower and DSpark checkpoint remain separate admission gates. No image-loaded timing, text-with-image-residency parity, draft acceptance, tool handling, logits, release, or installed build was qualified in this slice. The 4K prefill floor remains untested.

Use live teacher-forced captures from the exact pinned reference runtime, starting at the prelude and comparing through the first differing BF16 producer. Test paired ingress in both directions so expert arithmetic is held to the same input on each replay. Repeat the connected two-visit parity gate before extending the native worker. [Performance playbook](../PERFORMANCE_PLAYBOOK.md#inkling-2-mini-packed-experts)
