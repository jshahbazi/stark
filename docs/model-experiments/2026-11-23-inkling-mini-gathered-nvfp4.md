# 20261122-INKFASTPRE-07 · Inkling 2 Mini gathered NVFP4 prefill kernel for wide commands

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKFASTPRE-07` (prefill series registered 2026-11-22) / 2026-11-23 / 2026-11-23 |
| Kind | Kernel (prefill NVFP4 expert products) |
| Attempt state | Completed; the 8-bit paired timing was cut short |
| Result and acceptance criterion | Passed: outputs bit-identical to the previous kernel, replays hash-identical on both artifacts at 2,048- and 8,192-row commands, commit-check gates passed. Paired engine-only prefill at 8,800 tokens on the original artifact 1.130× (8/8). |
| Evidence scope | Kernel-level timing on a captured visit, full-model replays and self-consistency on both artifacts, paired engine-diagnostic timing on the original, partial pairs on the 8-bit artifact. No worker or server timing of this change alone. |
| Promotion decision | Default for packed prefill commands of 768 rows or more; `grouped_fast` stays below 768 rows; the exact reference is unaffected |
| Disposition | Committed in `4314ac51` (2026-11-23). Not released or installed. |
| Predecessor / related | [INKFASTPRE-03](2026-11-23-inkling-mini-prefill-round3.md#rejected-attempts) (the first MLX-Serve-structured port, not faster); [INKFASTPRE-04](2026-11-23-inkling-mini-prefill-round4.md) (expert ceiling, stationary kernels); [MLX-Serve source comparison](../benchmarks/2026-11-23-inkling-mini-mlx-serve-source.md); [differences page](inkling-mlx-serve-differences.md) |

Host as in the other 2026-11-23 records (128 GiB M5 Max, macOS 26.7.2, 94 W adapter, shared lock). Original checkpoint at `6e7c8248…` and the Stark 8-bit artifact (`c9e14848…`). Private outputs under `runtime/prefill-fast/` (`mlxceil`, `nax2`, `nax3`). Sources: the prefill workstream's reports and the commit body of `4314ac51`.

## Step 1: time MLX-Serve's own kernel on Stark's capture

`--expert-ceiling` gained an optional command-row count and an MLX-Serve kernel directory: it compiles MLX-Serve's `gather_qmm_nax` kernels from the pinned source (`runtime/third-party/mlx-serve-2fe9be83/`, revision `2fe9be83`) at run time with a Stark-written entry point and times them on one captured routed visit of an 8,192-row command (65,536 routes, original artifact, 10 repetitions per variant in one command). Their figures are the bare product with BF16 output; Stark's include the fused SwiGLU and route-weight epilogue. Their output values were not checked against Stark's (their weight-layout assumptions were not verified against Stark's bank); times only. GPU ms per visit, gate/up + down:

| Variant | Visit 2 | Visit 10 |
| --- | --- | --- |
| `grouped_fast` production | 22.85 + 13.97 = 36.8 | 27.51 + 15.66 = 43.2 |
| Stark tensor-only variant | 20.53 + 9.86 = 30.4 | 20.21 + 9.50 = 29.7 |
| Stark decode-only variant | 19.28 + 8.49 = 27.8 | 17.16 + 7.85 = 25.0 |
| MLX-Serve BM 96, GX 32 (their planner's pick) | 15.19 + 7.72 = 22.9 (46 TFLOPS) | 18.94 + 10.46 = 29.4 |
| MLX-Serve BM 64, GX 32 | 16.92 + 9.36 = 26.3 | 21.26 + 10.21 = 31.5 |
| MLX-Serve BM 64, GX 0 | 19.33 + 9.92 = 29.3 | 19.52 + 9.57 = 29.1 |

Their kernel, with decode and tensor work interleaved in the same SIMD groups, beat Stark's tensor-only variant, so overlap is reachable on this hardware and the gap was in the kernel, not in the decode. Their scan pre-pass cost 0.03 to 0.1 ms. Run-to-run variation was large; compare within a visit. Receipt `mlxceil/ceil-v2.json` SHA-256 prefix `3de95d75e7b54ade`.

## Step 2: a Stark kernel in that structure

`grouped_nax.metal` (replacing the round-3 port of the same name): a 96 × 64 tile of six 32 × 32 SIMD groups on 192 threads, 128 of them dequantizing a 64 × 64 weight block; double-buffered dequantized weights as the only threadgroup storage, with the next K step's packed words fetched into registers before the tensor work and dequantized into the other buffer after it, one barrier per K step; register-filled 16 × 32 × 16 tensor operations with activations loaded from device memory as four-element vectors; GX = 32 tile order and 96-row runs (`ink_group_bucket_nax_nvfp4`). Kept from Stark: the E2M1 × E4M3 decode of `experts.metal` and `grouped_fast.metal`, Stark's run table, and the fused gate/up activation and down-scale epilogues applied to the accumulator registers with the same arithmetic as `ink_group_gate_up` and `ink_group_down_epilogue`.

**Provenance.** A derivative of the `gather_db` schedule in mlx-serve `src/src/kernels/moe/gather_qmm_nax_header.metal` at `2fe9be834a601c70ef1a597563026340f0e722fe` (Apache-2.0), itself ported from oMLX `omlx/patches/m5_gather_qmm_nax.py` at `d6b2b92` (Apache-2.0) and adapted from MLX `steel/gemm/nax.h` (Copyright (c) 2025 Apple Inc., MIT). The file header (`SPDX-License-Identifier: Apache-2.0`) lists what was followed and what Stark changed; `provenance/UPSTREAM.md` records it.

**First build: 6 times slower.** Outputs were already 100 percent bit-identical to `grouped_fast` on visits 2 and 10, both projections (relative L1 0), but gate/up took 140 ms against 24 for `grouped_fast` (MLX-Serve's own kernel 13.4) and down 72 against 13. Cause: the fixed-size fragment loops did not unroll, putting the accumulator and fragment arrays on the stack; MLX-Serve forces full unrolling. Fixed by full unrolling of every fixed loop (`INK_NAX_UNROLL`).

**Unrolled build, per visit on the 8,192-row capture** (ms, two runs each):

| Visit | Projection | `grouped_fast` | New kernel | MLX-Serve BM 96 bare product |
| --- | --- | --- | --- | ---: |
| 2 | gate/up | 25.4 / 25.3 | 19.4 / 22.5 | 13.4 |
| 2 | down | 13.5 / 13.1 | 9.0 / 8.3 | 6.4 |
| 10 | gate/up | 27.2 / 29.2 | 22.3 / 23.4 | 16.3 |
| 10 | down | 12.8 / 13.1 | 8.5 / 10.1 | 9.9 |

20 to 35 percent faster per visit and bit-identical. The 16-deep register fragments accumulate exactly as the 32-deep tensor-view operations do, so the arithmetic class is unchanged. MLX-Serve's bare product remains about 6 ms faster on gate/up; how much of that is Stark's fused epilogue (two precise tanh, one exp, scattered BF16 stores) is not measured yet (plain-store diagnostic queued).

**What differed from the round-3 port that was not faster** (by the prefill workstream's account): the round-3 kernel used tensor views on device and threadgroup memory for both operands; this one fills register fragments with explicit vector activation loads from device memory, keeps only dequantized weights in threadgroup memory, uses 128 of 192 threads as loaders, and forces full unrolling. The round-3 result stands as a failure of that earlier design.

## Step 3: default selection and full-model checks

`grouped_pilot.rs`: `nax_nvfp4()` selects the new kernel when the command has at least 768 rows (MLX-Serve's rule of at least 48 mean rows per expert for K ≥ 1,024, with 128 experts and 8 routes per row); below that `grouped_fast` stays, so the 604-row tail of an 8,800-token prompt still uses `grouped_fast`. The expert-ceiling capture keeps the 64-row path. The exact reference never takes the fused path.

**Replays** (l8800, binary `nax3` with the kernel on against `nax2`, the same tree with the kernel unused): hash-identical on both artifacts at 2,048- and 8,192-row commands, snapshot and boundary replays exact in all eight runs. Original state `4d9c63e146e2`, seed logits `db19030a40f1`; 8-bit state `4d6c9ca36a00`, seed logits `1d6268c2e1b3`.

**Timing, engine-only, l8800:**

| Measurement | Before | After | Change |
| --- | ---: | ---: | --- |
| Original, paired ABBA, 8,192-row commands, four pairs (`nax3/pairs-orig.jsonl`, SHA-256 prefix `a2e173eb68d3cb8d`) | 3,895 ms (2,262 tok/s) | 3,346 ms (2,630 tok/s) | 1.130× median of pairs (1.125 to 1.206), 8/8 |
| Original, single runs, 2,048-row commands | 4,357 ms | 3,496 ms | Unpaired |
| Original, single runs, 8,192-row commands | 4,113 ms | 3,494 ms | Unpaired |
| 8-bit, single runs, 2,048-row commands | 4,158 ms | 3,536 ms | Unpaired |
| 8-bit, single runs, 8,192-row commands | 4,109 ms | 3,439 ms | Unpaired |
| 8-bit, paired ABBA, 8,192-row commands, cut by a pause (`nax3/pairs-a8.jsonl`, 13 of 16 runs) | 3,766 / 3,814 / 3,922 ms (pairs 0 to 2) | 3,337 / 3,346 / 3,484 ms | 1.129, 1.140, 1.126 for the three complete pairs (6/6 runs faster), computed for this record |

The commit body and the agent's report describe the 8-bit side as unpaired only; the receipt holds three complete ABBA pairs before the pause cut the fourth. Treat them as a partial paired result, not as the planned four pairs.

**Gates on the committed tree** (`runtime/commit-check-3/`, summary SHA-256 prefix `4d038aef3e5d6fc0`): clean build, 167 worker tests, exact reference exact on `g4096-02` and `s4096-00`, drafting on equals off at 73, 4,150, 8,800 and 16,000 on both artifacts, a repeated replay with identical hashes, the worker smoke on both artifacts.

## Not done

- Paired 8-bit timing with four complete pairs.
- A plain-store diagnostic for the fused epilogue's cost, to place the remaining gap to MLX-Serve's bare product.
- No worker or server timing of this change alone; the four-arm scoreboard on `4314ac51` was running when this record was written.
