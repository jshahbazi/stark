# 20261122-INKFASTPRE-01 · Exact faster prefill for Inkling 2 Mini: NVFP4 guard removal, fused NVFP4 kernels and faster MXFP8 replay

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKFASTPRE-01` / 2026-11-22 / 2026-11-22 |
| Kind | Kernel (prefill) with correctness checks |
| Attempt state | Completed. Its fourth planned step, an inexact MXFP8 path, failed the gate in force at the time and is the subject of [20261122-INKFASTPRE-02](2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md). |
| Result and acceptance criterion | Passed: bit-exact against the pinned reference on all nine captures at every step, and Stark self-consistency at 73 to 16,000 tokens. Paired engine-diagnostic timing: 1.506× at 4K and 1.476× at 8,800 tokens. |
| Evidence scope | Full-model numerical (four stock-schedule and five INKPREFILL-07 holdout captures), full-model self-consistency, paired engine-diagnostic timing. The worker was not timed by this attempt. |
| Promotion decision | Adopted as the default prefill path within the source-path scope. Superseded as the default by INKFASTPRE-02, which keeps these kernels and replaces the exact MXFP8 step. |
| Disposition | Adopted on branch, uncommitted on top of `f1666090`. Not merged, released or installed. |
| Predecessor / related attempts / PR | [Wider NVFP4 run tiles](2026-11-21-inkling-mini-nvfp4-run-tiles.md), [tensor MXFP8](2026-11-21-inkling-mini-mxfp8-tensor.md), [default fast prefill](2026-11-21-inkling-mini-prefill-default.md), [NVFP4 guard attribution](2026-11-21-inkling-mini-nvfp4-router.md), [20261122-INKDEFAULT-01](2026-11-22-inkling-mini-default-fast-paths.md). Successor: [20261122-INKFASTPRE-02](2026-11-22-inkling-mini-unguarded-mxfp8-prefill.md). No PR. |

## Change

Three exact steps, each measured against the previous one, plus one inexact step that was not landed:

1. **NVFP4 guard and replay removed.** The per-K-step bound checks (SIMD max, min and shuffle), the replay loop and the `quantum` and `maxima` guard inputs computed at ingress are gone. The plain tensor product remains, with a vectorized F32-to-BF16 gather at ingress (`INK_GROUP_QMM_BARE` build of the product, new `ink_group_ingress_bare`). On every checked capture the guard never changed a result; the report's explanation is that NVFP4 products have so few significant bits that F32 sums are exact in practice. That explanation was not proved separately.
2. **Unguarded MXFP8 tensor path.** Tried and not landed; see below and INKFASTPRE-02.
3. **NVFP4 kernel geometry and fused epilogues.** New `ink_group_fast_gate_up` and `ink_group_fast_down` in `grouped_fast.metal` replace the 64 × 32 product, the separate `ink_group_gate_up` and `ink_group_down_epilogue` kernels and their F32 round trip through memory. They use 64 × 64 tiles with 32 × 32 SIMD-group fragments, BK 32, a vectorized and double-buffered weight decode, and apply the source gate/up activation and route weight, or the down scale, at the store, so BF16 leaves the kernel directly.
4. **Faster exact MXFP8 replay.** The certificate and source-order replay are kept. The replay reads the sorted BF16 ingress rows instead of gathering F32 rows, with identical arithmetic; rows without an exact BF16 form fall back to the F32 path.

During the self-consistency checks the first build sent verifier rows through the fused NVFP4 path, which broke the rollback proof. The fast path now excludes verifier commands, so the decode and verifier kernels are untouched. The prefill code path did not change with that fix, so the timings and parity results apply to the fixed build.

Source, all under `crates/stark-engine/`: `grouped_fast.metal` (new), `grouped_pilot.metal`, `grouped_pilot.rs`, `mxfp8_tensor.metal`, `mod.rs`, `schedule_validation.rs`, `prefill_profile.rs`, `session.rs` and `examples/inkling_slice.rs`.

## Artifact, source and environment

| Field | Value |
| --- | --- |
| Checkpoint | Original `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, hashes as in the [default-router record](2026-11-21-inkling-mini-decode-default.md#artifact-and-run-controls) |
| Reference | Pinned stock-schedule Ollama captures (`g4096-01`, `g4096-02`, `s4096-00`, `s4096-01`) and the five INKPREFILL-07 holdouts ([capture protocol](../archive/experiments/2026-11-21-inkling-oracle/README.md), [holdouts](2026-11-21-inkling-mini-packed-prefill.md#20261121-inkprefill-07--source-precision-holdouts)) |
| Source | Uncommitted working tree on top of `f1666090`; no patch hash recorded |
| Baseline build | The engine diagnostic before step 1; its hash is not recorded in the report |
| Step builds (`inkling_slice` SHA-256) | Step 1 `ac0670074a36a8e1f8593f7b902be4235fd66b394b7ba37105afa3e1cf301d5f`; step 3 `739eae247aeb900931c5d8f9d01a7b49ec5b12377e9dbc280799e7a087db4640`; final with the verifier exclusion `a92cc9ee00136e3b8bd2f31381b4e8a971ad82f112001eccc9e28028c07c946a` |
| Matching worker | `stark-generate` `73a0fa2e1a5191e1087f2a7152d72b5e280466e2d8e06a5802c8adc80672e76b` from the same tree state, used by [INKENGINE-02](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md#20261122-inkengine-02--second-run-with-the-author-8-bit-pack) and the [prefill attribution](../benchmarks/2026-11-22-inkling-mini-prefill-attribution.md); not timed by this attempt |
| Precision | Unchanged from the exact source path: NVFP4 products with F32 accumulation and BF16 stores, certified MXFP8 with source-order replay |
| Host | 128 GiB Apple M5 Max (Mac17,7), macOS 26.7.2 (25G332), AC power; host shared with other agents' GPU jobs, serialized through one lock |

## Correctness

| Check | Result |
| --- | --- |
| Pinned parity after steps 1, 3 and 4 | Bit-exact on all nine captures: relative logit L2 0 on every row, 4/4 top-1, zero differing residual elements per visit, state equal to Stark's exact path |
| 4K state, seed logits and residuals against the baseline build | Exact |
| `--measure-verify-transaction-schedule` at N = 73, 4,150, 8,800 and 16,000 | Pass: serial match at every depth, rejection and crossing controls, abandoned-transaction reset, cross-mode snapshot rejection; boundary replays at none, 4,096, 4,096/8,192 and 4,096/8,192/14,336. At 4,150 the seed logits, serial teacher and serial state hashes equal the baseline binary. |
| `--schedule-replay` at 73, 4,150, 8,800 and 16,000 | Exact snapshot and boundary replays |
| Run-to-run determinism | Identical state and logit hashes on two runs each at 73, 4,150 and 16,000, and on all eight runs per arm in the timing pairs |
| Independent re-check | The coordinating session re-verified parity on its own; its outputs are not bound in this record |

## Performance

ABBA process pairs with the same inputs on the engine diagnostic, held under one lock. 4K used the four stock captures, one per pair; 8,800 used the l8800 tokens for all four pairs. Medians with ranges.

| Comparison | 4K | 8,800 tokens, context 9,216 |
| --- | --- | --- |
| Step 1 against baseline | 928 → 1,213 tok/s, 1.325× (1.294–1.333), 4/4 wins | 963 → 1,258 tok/s, 1.305× (1.300–1.311), 2/2 wins |
| Step 3 against step 1 | 1.080× (1.077–1.093), 4/4 | 1.081× (1.080–1.082), 2/2 |
| Final against step 3 | 1.049× (1.043–1.058), 4/4 | 1.049× (1.048–1.051), 2/2 |
| Final against baseline | 1,085 (1,081–1,096) → 1,634 tok/s (1,633–1,642), **1.506×** (1.493–1.511), 8/8 | 1,034 (889–1,068) → 1,455 tok/s (1,231–1,603), **1.476×** (1.244–1.541), 8/8 |

Absolute times drifted upward by as much as 20 to 30 percent during long sessions on this host, while ratios stayed stable across pairs and launch orders. Compare ratios, not absolute numbers from separate sessions.

Instrumented 4K stage changes (GPU ms): step 1 gate/up dot 1,081 → 738, down dot 940 → 321, whole command 3,979 → 2,974; step 3 gate/up 738 → 614, down 321 → 297, epilogue 83 → 14; step 4 MXFP8 replay about 550 → 371.

Remaining instrumented GPU time on the final build: 4K 2.59 to 2.76 s, of which NVFP4 dot products 34%, dense attention projections 21%, MXFP8 20% (replay 0.37 s), relative and cache attention 9%, NVFP4 routing 6%, attention and expert inputs 6%, other 6%, quantization 2%. At 8,800 tokens 5.85 s, with similar shares and global attention 0.31 s.

## Rejected and failed variants

| Variant | Outcome |
| --- | --- |
| NVFP4 BK 64, with or without double buffering | Slower: too much threadgroup memory |
| NVFP4 128 × 32 and 128-row tiles | Slower |
| Eight-row and row-major MXFP8 replay | Slower; removed |
| Unguarded MXFP8 tensor path (one F32 accumulation over K, BF16 store, no TwoSum, certificate masks or replay) | Saved about 0.5 s at 4K. Failed the INKPREFILL-07 gate on 4 of 9 inputs (stock `g01`, `g02`, holdouts `g01`, `g02`; mean row L2 1.26% to 5.17%, worst row 16.0%, top-1 3/4 on two inputs). Not landed in this attempt. Its re-evaluation under a noise-floor gate is INKFASTPRE-02. |

Kernel ablation found decode and tensor work each about half of the fused NVFP4 kernel without overlap; the vectorized decode was the main gain.

## Tooling

`--prefill-tolerance` (default prefill against the bit-exact reference on one loaded target, per-row L2, top-1, unequal logits and differing residuals per visit, INKPREFILL-07 gate), `--prefill-tolerance-mxfp8` (the inexact candidate, diagnostic only) and `--profile-schedule` (stage profile at any prompt length). `--profile-stock-4k` now uses the exact kernels as its control. The exact reference mode `Runtime::exact_prefill` is set only by diagnostics. The stale `attention` and `trace.mlp_conv` probes were not fixed here (see [INKFUSE-02](2026-11-22-inkling-mini-launch-fusion.md#20261122-inkfuse-02--second-session-paired-timing-and-single-row-fusions)).

## Limits

- Only the engine diagnostic was timed; the worker effect of this change alone was not measured. The worker's later engine-matrix and attribution figures use a build from the same tree state.
- Planned items not done: mask-free global attention (done later under INKFASTPRE-02) and 4,096-row commands, which would need a larger workspace than the worker's 1,344 MiB reservation and coarser prefix snapshots.
- The private run outputs are under the ignored `runtime/prefill-fast/` directory; their hashes are not recorded here.
