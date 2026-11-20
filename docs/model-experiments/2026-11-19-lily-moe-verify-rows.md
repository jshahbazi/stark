# 20261119-LILYROWS-03: share routed experts across verify rows

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | 20261119-LILYROWS-03; registered and executed 2026-11-19; workstream `moe-rows` of the Lily MTP speed project |
| Kind | Kernel (performance), with an overlap diagnostic |
| Attempt state | Completed |
| Result and acceptance criterion | Stop rule passed (cross-row overlap above 20 percent at 3 rows). Exact sharing kernels built and byte-identical. Performance: not faster (gate/up even at best, down/combine slower). Rejected for the default path. |
| Evidence scope | Operator/component byte tests and timings; full-model numerical (output-hash gate, MTP rollback test); end-to-end staged screening |
| Promotion decision | Rejected. Not faster; the default-off kernels and selector were not merged |
| Disposition | The code is not on the integration working tree. The source patch `moe-rows.patch` is archived as unadopted experimental source in [`docs/archive/experiments/2026-11-19-lily-verify-rows/`](../archive/experiments/2026-11-19-lily-verify-rows/README.md); not committed, released or installed |
| Predecessor / related attempts / governing issue / PR | Baseline 20261119-SUSHI-02; precedent 20261116-LILY-REUSE-01 (projection reuse). No PR. |

## Hypothesis and change

- Problem: routed experts cost about 2.6 ms of GPU time per extra verify row (gate/up 1.4, down/combine 1.05). Each row reads its own 10 experts, and an expert that several rows chose is read again. If the stages are bound by DRAM bandwidth, reading each such expert once per pass should cut the slope roughly in proportion to the duplicate share.
- Baseline: `stark_moe_gate_up_decode_rows` (grid rows x 640 x top_k, 32 lanes per output channel) and `stark_moe_down_combine_inject_decode_rows` (grid rows x 1280, 128 threads, 16 lanes per output row, four simdgroups).
- Candidate: `stark_moe_gate_up_shared_rows_r{4,9}` (grid 640 x rows*top_k; a threadgroup whose entry is the first selection of its expert serves every row that chose that expert, and the rest exit) and `stark_moe_down_combine_shared_rows_r{4,9}` (one threadgroup per output-row pair for all rows; it builds an ownership table in threadgroup memory, reads each distinct expert once, and runs the per-row slot-ordered combine and HC write). Every row keeps the ordinary decode body, lane mapping, reduction and BF16 boundaries. The row count is a template parameter (4 and 9 are compiled).
- Correctness fixed in advance: byte equality with the per-row kernels and with ordinary decode; baseline output hashes for every arm.
- Stop rule fixed in advance: if fewer than 20 percent of selections are duplicates at 3 rows, record the result and do not build.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` rev `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | same |
| Format/quantization | Q4 group 64 affine experts (512 experts, top-10, 48 layers); router, shared expert and shared gate are Q8 | same |
| Backend, vendor revision and local patch provenance | vendored lily-core at Stark `4d09b550` | Stark-local patch: `kernels/metal/stark_moe_shared_rows.metal` (new), `kernels/moe.rs`, `moe_ffn.rs`, `moe_row_overlap.rs` (new, diagnostic), `moe_decode_exact_tests.rs`, and an 11-line diagnostic hook in `qwen4exp/model.rs`. Original Stark code, nothing adapted from Sushi. |
| Stark commit, dirty patch and executable hash | `4d09b550`, worker `5dfb90b6...298d` (20261119-SUSHI-02) | `4d09b550` plus `runtime/patches/moe-rows.patch`; diagnostic worker `6f13bddf...baf3f`, kernel worker `c707d4f2...f068` |
| Toolchains, dependency lock hashes and build flags | rustc 1.98.1 release (worker); nightly-2026-10-01 (tests); Metal 32023.883; Cargo.lock `7e6dfa20...066c` | same |
| Weight, activation, accumulation, recurrent-state and KV precision | unchanged | unchanged (same per-row arithmetic) |
| Selected kernel and materialization policies | per-row expert kernels | default unchanged; `STARK_LILY_MOE_SHARED_ROWS=gate_up` or `all` selects the shared kernels |
| Hardware, physical RAM, OS/driver/Metal toolchain | Apple M5 Max, 128 GB, macOS 26.7.2 | same |
| Available headroom, observed peak memory and measurement method | not measured | not measured (the shared kernels add no buffers; the diagnostic allocates 64 KB only when enabled) |

## Workload and controls

- Image tower loaded (the runner's default native launch); no image requests. Image and MTP parity N/A: this change affects only verify-row MoE dispatch, and the default path is unchanged.
- Corpus: the frozen public code corpus at 4,096 input tokens; the input hash matches the baseline. Arms: sampled (temperature 1, top-k 20, top-p 0.95, seed 42, 256 tokens) and greedy (512 tokens) at 0 to 3 drafts. Fresh requests, one repeat. Every real-model run went through the model-slot wrapper.
- Host not quiet: other workstreams shared the GPU and CPU, so all timings are screening only.

## Step 1: bandwidth and overlap

Tensor-header bytes per expert per layer: gate, up and down codes at 819,200 each, plus 51,200 scales and 51,200 biases each, giving 2,764,800 bytes. A row reads 10 x 48 of them, 1.327 GB per pass (gate/up 0.885 GB, down 0.442 GB). That is slightly above the brief's 1.18 GB estimate, because the brief left out scales and biases.

Overlap counts from the private diagnostic (`STARK_LILY_MOE_ROW_OVERLAP_OUTPUT`, counts only, run `moe-rows-1`):

| Arm (rows) | Verify passes | Duplicate selections | Saved reads (1 - distinct/selections) | New experts per extra row | Upper-bound bytes saved per extra row |
| --- | ---: | ---: | ---: | --- | ---: |
| s1 (2) | 146 | 34.2% | 17.1% | 0.658 | 0.45 GB |
| s2 (3) | 110 | 44.3% | 24.7% | 0.667, 0.594 | 0.49 GB |
| s3 (4) | 102 | 50.3% | 29.7% | 0.673, 0.594, 0.544 | 0.53 GB |
| g2 (3) | 183 | 38.7% | 21.4% | 0.745, 0.613 | 0.43 GB |
| g3 (4) | 147 | 45.9% | 26.9% | 0.726, 0.620, 0.579 | 0.48 GB |

The stop rule passed (s2 44.3 percent, g2 38.7 percent). If the stages were purely bandwidth-bound, the slope would be predicted to fall from about 2.45 to about 1.6 ms per extra row (about 35 percent fewer bytes per extra row). The diagnostic run itself reproduced every baseline output hash.

Bandwidth check (component test `moe_rows_expert_overlap_timing`: 48 distinct synthetic layers of 40 experts, zero weights, per-row kernels, median of 12; ms for 48 layers):

| Rows | gate/up distinct | partial | same | down distinct | partial | same |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.67 | | | 1.07 | | |
| 2 | 3.16 | 2.78 | 2.07 | 1.92 | 1.87 | 1.85 |
| 3 | 4.60 | 3.95 | 3.01 | 2.78 | 2.71 | 2.69 |
| 4 | 6.05 | 5.18 | 3.95 | 3.64 | 3.55 | 3.53 |

What the table shows:

- Gate/up with distinct experts costs about 1.45 ms per extra row (about 610 GB/s, close to DRAM peak). With every row choosing the same experts, so that only one row's bytes come from DRAM, it still costs about 0.95 ms per extra row. That remainder is per-row arithmetic: nibble decode, conversion and two dot products for every weight and every row. The existing per-row kernel already gets part of the duplicate saving from the cache ("partial" falls between "distinct" and "same").
- Down/combine costs about 0.85 ms per extra row whether rows share experts or not. It is not bound by DRAM bytes, so sharing reads cannot help it.
- Upper bound on any exact sharing kernel that is as efficient as the per-row kernel: about 0.3 to 0.4 ms per extra row on gate/up (partial to same) and about 0 on down. That is far below the 0.85 ms predicted from bytes alone.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| Overlap diagnostic `moe-rows-1` | Counts per verify pass, aggregated per request | 34 to 50 percent duplicate selections; hashes equal to baseline | Valid; diagnostic kept default-off |
| Bandwidth control | Per-row kernels over distinct, partial and same routes | Gate/up per extra row is about 0.95 ms ALU-bound and about 1.45 ms DRAM-bound; down is not DRAM-bound | Valid component screening, other agents active |
| Shared kernels v1 | Shared weight words and affine terms; each row runs the full decode block body in turn | Byte-identical. Gate/up: 2 rows partial 2.76 to 3.06 ms (slower), 3 rows 4.21 to 4.13, 4 rows 5.63 to 5.44 (within noise); same pattern slower at every row count. Down: slower in every case (3 rows partial 2.72 to 3.39) | Exact, not faster; rejected as default |
| Shared kernels v2 | Nibble decode hoisted out of the row loop to cut per-row arithmetic | Bytes changed (first mismatch at rows 1, pattern none: the default math mode compiles the reordered code differently) and much slower (register arrays) | Invalid under the exactness contract; reverted |
| Real-model staged A/B `moe-rows-3/4/5` (s1, s2, s3; off / gate_up / all) | Same binary, private selector | Verify GPU command at 2/3/4 rows: off 20.57/26.84/34.90, all 21.14/27.32/34.07. HC (not touched by this change) varied from 4.05 to 6.56 ms across runs | Too noisy to resolve under 1 ms; consistent with the component tie. Inconclusive at best |
| Hash gate `moe-rows-2` (all arms, selector `all`) | End-to-end exactness | All 8 output hashes and work counts equal to baseline | Valid for exactness. Its timings are invalid: an engine test build ran concurrently (s0 also fell 10 percent) |

## Correctness and lifecycle

- Component byte tests (physical Metal, release, nightly-2026-10-01), all passing:
  - `expert_shared_rows_match_per_row_bytes`: 4 independent draws (two ordinary; one with subnormal-heavy inputs, scales and biases; one with large cancelling inputs and biases) x rows 1 to 9 x 6 route patterns (none, all, all permuted, some, pairs, random). Covers gate/up, and down with and without the HC write, with output canaries. 432 down cases, 52,531,200 bytes compared, 0 unequal. A first version with one ordinary draw also passed (108 cases).
  - Residual risk, reported by the HC workstream: under fast math, the GPU back end can schedule identical per-row code differently depending on the surrounding kernel. That produced rare single-BF16 flips in another workstream's shared-weight kernel that passed one draw. The draws above did not reproduce it here, but byte identity is shown by sampling, not proven. That adds to the rejection.
  - `decode_exact_moe_rows_match_ordinary_bytes`: extended to 16 and 64 experts, shared modes off, gate_up and all, rows 1 to 4, both router renorm settings.
  - `batched_moe_injection_matches_ordinary_fused_bytes` (72 cases) and `official_decode_exact_moe_rows_match_ordinary_bytes` still pass.
- Real-model: the end-to-end hash gate passes with the shared kernels on. The `private_checkpoint_mtp_verify_and_rollback_match_serial` result is in the follow-up below.
- Prefix reuse and rollback: unaffected (no state, scratch or cache change).

## Performance and memory

| Metric and measurement boundary | Baseline | Candidate | Repeats/dispersion | Eligibility and conclusion |
| --- | --- | --- | --- | --- |
| Gate/up, 3 rows, partial route (component, 48 layers) | 4.21 ms | 4.13 ms | median of 12, noisy host | tie |
| Down/combine, 3 rows, partial route (component) | 2.72 ms | 3.39 ms | median of 12 | regression |
| Verify GPU command, s2 (real model, staged) | 26.84 ms | 27.32 ms (all) | one run each | inconclusive, contended host |
| Useful decode / end-to-end rate | not measured validly | not measured validly | | invalid run (concurrent build) |
| Peak memory | not measured | not measured | | no new buffers |

## Evidence, decision and limits

- Cross-row expert overlap is large (17 to 30 percent of reads, 34 to 50 percent of selections), but the routed-expert slope is not mainly DRAM bytes. Gate/up per extra row is bound by the exact per-row arithmetic at about 0.95 ms even with perfect sharing, and the existing kernel already gets part of the saving from the cache. Down/combine does not depend on bytes at all.
- Exact sharing therefore leaves about 0.3 to 0.4 ms per extra row to gain on gate/up at most, and the exact shared kernel built here does not capture it. Cutting per-row arithmetic (decoding once for all rows) changed bytes under the default math mode.
- Decision: do not adopt. The selector and kernels are default-off; recommend removing them, or keeping them only as an archived reference.
- The diagnostic (`moe_row_overlap`) is default-off and costs one environment check per verify layer when off. Keep it only if overlap needs measuring for the 9-row depth; otherwise remove it with the kernels.
- Rest of the MoE verify path: router (Q8, F32 output), shared gate (Q8), shared expert gate/up and down (Q8 BF16) all take the projection-reuse path and read weights once. The routed experts are the only per-row weight reads left.
- Next: a lower-arithmetic exact formulation would need to match decode bytes. Possible routes are an explicit `fma`/precise math-mode decode kernel with a matching verify kernel (this changes the decode reference, so it needs its own precision attempt), or a different row-parallel lane split. Down/combine latency (about 0.85 ms per row, not bytes) is a separate target: dispatch shape, or fusion with gate/up.

## Follow-up history

- 2026-11-19: `private_checkpoint_mtp_verify_and_rollback_match_serial` (needs `STARK_TEST_MODEL_PATH`; the shared brief first named it `STARK_LILY_TEST_MODEL_PATH`) passed on the real model with `STARK_LILY_MOE_SHARED_ROWS=all`. The verify (86 readouts), rollback (24) and host-serial (30) phases have 0 unequal values. The draft-head phase has 248,320 unequal values, a known allowed difference with 0 top-1 mismatches; this change does not touch the draft head.
- 2026-11-19: the byte test was strengthened to 4 independent draws, including subnormal-heavy and cancellation inputs (see Correctness), after the HC workstream reported fast-math scheduling flips. Still 0 unequal.
- 2026-11-19, integration: rejected as recommended. The kernels, selector and overlap diagnostic are not on the integration working tree. The source patch `moe-rows.patch` (SHA-256 `49235c95fb57957830966f80efbc4ef6160a81f46f63d5881321c2963d8402cf`) is archived unapplied in [`docs/archive/experiments/2026-11-19-lily-verify-rows/`](../archive/experiments/2026-11-19-lily-verify-rows/README.md).
