# 20261122-INKFASTPRE-02 · Unguarded MXFP8 prefill as the Inkling 2 Mini default, under a noise-floor gate

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKFASTPRE-02` / 2026-11-22 / 2026-11-22 to 2026-11-23 |
| Kind | Precision change (prefill MXFP8 accumulation), new correctness gate, exact kernel (mask-free global attention) |
| Attempt state | Completed |
| Result and acceptance criterion | Passed both conditions set before the run: (1) Stark self-consistency, (2) the new noise-floor gate. The candidate is not bit-exact against the pinned reference and fails the earlier INKPREFILL-07 gate, which is kept for reference only. Paired engine-diagnostic timing: 1.169× at 4K and 1.174× at 8,800 tokens over the exact build. |
| Evidence scope | Full-model numerical against nine pinned captures with a perturbation control, full-model self-consistency, row-independence check, paired engine-diagnostic timing. No worker timing of this build. |
| Promotion decision | Promoted to the default prefill path within the source-path scope. The bit-exact kernels remain reachable only through the `--exact-prefill` diagnostic. |
| Disposition | Adopted on branch, uncommitted on top of `f1666090`. Not merged, released or installed. |
| Predecessor / related attempts / PR | [20261122-INKFASTPRE-01](2026-11-22-inkling-mini-exact-fast-prefill.md) (exact kernels and the first failed evaluation of this candidate); [INKPREFILL-07 gate](2026-11-21-inkling-mini-packed-prefill.md#20261121-inkprefill-07--source-precision-holdouts); [compensated tensor MXFP8](2026-11-21-inkling-mini-mxfp8-tensor.md). No PR. |

## Why the gate changed

The INKPREFILL-07 gate required each candidate input to keep the incumbent's greedy top-1 agreement, to keep mean reference-relative logit L2 within two percentage points of the incumbent, and to add no more than five points on any teacher row. It was set when the incumbent itself had 10 to 40 percent error against the pinned reference. Once the incumbent became bit-exact (INKFASTPRE-01), the same gate compares any inexact candidate with zero, and a single changed BF16 bit anywhere propagates through the model. The unguarded MXFP8 candidate failed it on 4 of 9 inputs.

To decide whether that failure measures the candidate or the propagation, a control was run first: the exact kernels with one residual element raised by one BF16 ulp at one visit. Twelve trials per capture over nine captures (four stock, five holdouts), at visits 1, 4, 8, 12, 17, 22, 27, 32, 37, 41, 45 and 47, with positions spread across the packed prompt rows and varied elements.

| Arm | Mean row L2 | Worst row L2 | Top-1 flip rate |
| --- | ---: | ---: | ---: |
| Control trials, range across trials | 0.00–18.6% | 0.00–87.3% | 0–30.6% |
| Control, median of trial means | 10.3% | — | — |
| Unguarded MXFP8 candidate | 2.38% | 16.0% | 5.6% |
| Exact reference | 0 | 0 | 0 |

Per-trial aggregates across the nine captures, mean / worst / flip rate: trial 0 18.6 / 87.3 / 30.6%; 1 17.1 / 69.0 / 25.0%; 2 17.7 / 86.3 / 30.6%; 3 10.3 / 45.8 / 16.7%; 4 11.7 / 34.0 / 13.9%; 5 10.3 / 30.9 / 27.8%; 6 10.7 / 32.0 / 13.9%; 7 9.3 / 29.4 / 8.3%; 8 8.1 / 23.0 / 19.4%; 9 3.9 / 20.1 / 8.3%; 10 0.27 / 0.82 / 0%; 11 0 / 0 / 0. Trials 10 and 11 perturbed rows near the end of the prompt and barely spread.

Row L2 quantiles (minimum, 10%, 25%, median, 75%, 90%, maximum): controls 0, 0, 2.5, 7.7, 13.1, 21.5 and 87.3%; candidate 0.5, 0.9, 1.4, 1.7, 2.0, 2.6 and 16.0%.

**The noise-floor gate** (`noise_floor_pass` in `--prefill-tolerance`): the candidate's mean row L2, worst row L2 and top-1 flip rate must each lie inside the span of the control trials, and its mean must be at most twice the control median. The candidate passes: all three aggregates are inside the span and its mean is about a quarter of the control median. One harmless ulp flip moves the model more than the whole unguarded MXFP8 path does on these inputs.

This gate decides whether a change is distinguishable from the model's own sensitivity to a single rounding difference. It does not show output quality, and it is calibrated on nine 4K captures only.

## Change

- **Unguarded tensor MXFP8 in prefill.** One F32 tensor accumulation over K with BF16 rounding at the store, with no TwoSum, certificate masks or replay. No environment selector.
- **Mask-free global attention**, bit-identical to the mask path on all nine captures. A new [`global_relative.metal`](../../crates/stark-engine/src/engine/inkling_slice/global_relative.metal) computes the F32 relative bias in the kernel from the BF16 relative table and sink, the same value `ink_global_probe_mask` writes, instead of reading an explicit F32 mask tensor. Every other operation is the upstream kernel's.
- **`--exact-prefill`** on `--full-target`, `--full-target-batched`, `--measure-verify-transaction`, `--full-target-schedule`, `--schedule-replay`, `--measure-verify-transaction-schedule` and `--measure-source-dspark-depth-schedule`. It is a function parameter only; no environment variable, and nothing the worker can reach. Reports include `exact_prefill_reference`.
- **Diagnostics:** `--prefill-tolerance <target> <context> <trials> <captures...>` with the noise-floor gate and `mask_free_global_identical`, the old gate under `former_inkprefill07`; the `ink_packed_perturb` kernel and the `prefill_perturb` and `global_mask_reference` runtime fields; `--split-consistency`; `--profile-schedule-draft`.

Source, under `crates/stark-engine/`: `grouped_pilot.rs`, `schedule_validation.rs`, `prefill_profile.rs`, `transaction_validation.rs`, `target_validation.rs`, `packed_global_probe.rs`, `packed_residual.metal`, `packed_prefill.rs`, `mod.rs`, `session.rs`, `global_relative.metal` (new), `metal_reference/attention.metal` (one include line), `build/inkling_reference.rs` (a rerun hint) and `examples/inkling_slice.rs`.

### Provenance of `global_relative.metal`

The kernel is a derivative of `attention_nax` in `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h` at MLX revision `fc344c986c4169d99f76b5d95ae6b4a58a359a63`, Copyright 2024-26 Apple Inc., MIT licence. The file carries `SPDX-License-Identifier: MIT`, names the upstream file and revision, and points to `metal_reference/LICENSE`. It is included at the end of `metal_reference/attention.metal`, which supplies the pinned attention header, so both kernels share one translation unit. The pinned upstream `mlx/` tree under `metal_reference/` is unchanged.

## Artifact, source and environment

| Field | Value |
| --- | --- |
| Checkpoint | Original `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, hashes as in the [default-router record](2026-11-21-inkling-mini-decode-default.md#artifact-and-run-controls) |
| Reference | The four stock-schedule captures and five INKPREFILL-07 holdouts |
| Baseline build | Exact build from INKFASTPRE-01, `inkling_slice` `a92cc9ee00136e3b8bd2f31381b4e8a971ad82f112001eccc9e28028c07c946a` |
| Candidate build | `inkling_slice` `60c51ea41b47a8943c081d6705f38df01249dd7b06c2b0e591135bd5236a3d69`. No matching `stark-generate` hash is recorded. |
| Source | Uncommitted working tree on top of `f1666090`; no patch hash recorded |
| Precision policy | Prefill MXFP8 visits: F32 tensor accumulation over K, BF16 store, no compensation or certified replay. Everything else as in the exact source path. Whether decode or verifier commands use the unguarded path is not stated in the report; INKFASTPRE-01 excluded verifier commands from its fast kernels. |
| Host | 128 GiB Apple M5 Max (Mac17,7), macOS 26.7.2 (25G332), AC power; every build and run through the shared GPU lock |
| Private aggregates | `runtime/prefill-fast/phase2/tolerance.json`, SHA-256 `9c4e12fada093585aa1069b2434cd03a4a2f51d2c41c7440e1f95eea3cefa177` |

## Correctness

**Condition 1, self-consistency (passed):**

- `--measure-verify-transaction-schedule` at 73, 4,150, 8,800 and 16,000 tokens: serial match at depths 1 to 3 with exact rejection and crossing controls, abandoned-transaction reset and cross-mode snapshot rejection; exact boundary replays at 4,096 (4,150), 4,096 and 8,192 (8,800), and 4,096, 8,192 and 14,336 (16,000).
- `--schedule-replay`, two runs at each of the four lengths: exact snapshot and boundary replays; identical prefill state, logits and per-row cache hashes between runs.
- Row independence (`--split-consistency`): the first 6,000 rows of l8800, b12000 and l16000, each prefilled under four command splits (1,200-row commands; alternating 1,100 and 1,900; 1,500-row commands; 1,777, 1,234, 1,777, 1,212) give bit-identical state and next-token logits, for the default and the exact kernels alike.

The first version of the split check mixed widths from 333 to 1,500 rows and failed for the exact path too. The source arithmetic switches by command width: the split-K router and dense reduction split change at 1,024 rows, and the thin projection tiling at 2,048 rows. A row's state therefore depends on whether its command was at most 1,024, 1,025 to 2,047, or 2,048 rows wide. This was already true of the exact reference. Prefix reuse stays exact because snapshots sit at 2,048-row boundaries and the remainder replays the same schedule. The final split check keeps every chunk between 1,025 and 2,047 rows.

**Condition 2, noise-floor gate (passed):** see above.

**The exact reference still reproduces the previous exact build** (same commands on the INKFASTPRE-01 build):

- `--full-target-schedule --exact-prefill` on the four stock captures: `all_logits_exact` true; cache, per-row logits and taps hashes equal.
- `--full-target-batched --exact-prefill`: zero unequal logits on all four rows; cache and logits equal.
- `--measure-verify-transaction --exact-prefill` at 4,096: no difference in any of 1,424 report fields other than timing.
- `--measure-verify-transaction-schedule --exact-prefill` at 4,150: seed logits, serial teacher and serial state hashes equal.

Without the option, `--full-target-batched` reports 200,058 unequal logits per row and top-1 agreement on 3 of 4 rows. That is the expected signature of the inexact default against the pinned reference.

## Performance

ABBA process pairs on the engine diagnostic, held under one lock per batch, against the exact build:

| Prompt | Exact build | Unguarded default | Paired speedup | Wins |
| --- | ---: | ---: | ---: | ---: |
| 4K | 1,623 tok/s | **1,903 tok/s** | 1.169× (1.107–1.184) | 8/8 |
| 8,800 tokens | 1,566 tok/s | **1,854 tok/s** | 1.174× (0.907–1.241) | 7/8 |

The losing 8,800 pair had outliers in both arms (7.6 s and 7.1 s). Against the original baseline of INKFASTPRE-01 (about 1,085 and 1,034 tok/s in that session), the totals are about 1.75× at 4K and 1.8× at 8,800; these cross-session totals are indicative only.

Instrumented GPU time at 8,800 tokens on the candidate totals 5.33 s: NVFP4 dot products 2.10 s, dense attention projections 1.29 s, relative and cache attention 0.65 s (local 0.25, global 0.28), NVFP4 routing 0.39 s, other 0.38 s, attention and expert input preparation 0.35 s, MXFP8 0.34 s (1.31 s on the exact path), NVFP4 quantization 0.14 s. Global attention went from 365 to 278 ms in the instrumented profile with the mask-free kernel.

Drafting in prefill: with drafting on, prefill encodes the draft commit serially per row. At 8,800 tokens it added about 650 ms of instrumented GPU time ("other" 383 → 1,030 ms), while uninstrumented wall time moved from 5,776 to 5,825 ms in one sample each. See the [prefill attribution](../benchmarks/2026-11-22-inkling-mini-prefill-attribution.md) for the worker-level cost.

## Failed, skipped or not done

- The first consistency run failed on the author's own bug (a Metal parameter named `thread` in the perturbation kernel); results above are from the fixed binary.
- `cargo test --lib` does not compile on Rust 1.98.1 in this crate (a `#![feature]` attribute in `lib.rs`, errors in `glm_metal.rs` and an existing `schedule_validation` test), so the new `split_ranges` unit test did not run.
- The explicit global mask buffer is still allocated (up to 957 MiB at 32K) because the exact reference uses it; freeing it needs a change to the worker's admission figures.
- No worker timing, image-loaded run or prefix-restore run through `stark-generate` on this build.
- The noise-floor gate is calibrated on nine 4K captures; no capture exists at other lengths.

## Next

Under the noise-floor gate, larger tensor tiles for the dense projections become admissible (estimated 0.4 to 0.6 s at 8,800 tokens, not measured), then the routing chain. Record each as its own attempt.

## Follow-up history

- 2026-11-23, [20261122-INKFASTPRE-03](2026-11-23-inkling-mini-prefill-round3.md): the dense projection kernel landed under this gate (mean row L2 15.2 percent, above the control median but inside the gate), a fixed-split router failed it, and the explicit global mask is now freed on the default path.
- 2026-11-23, limit found after this record: the nine captures are synthetic, nearly uniform prompts, on which the model's top-1 matched the actual next token 0 times in 34 on one input. The control's and candidate's top-1 flip rates above therefore measure sensitivity on that text, not quality on real text. The gate stays valid as a sensitivity comparison and the self-consistency checks as exactness checks. A natural-text comparison of default against exact prefill has not been measured.
- 2026-11-23, natural text measured on a later default (this attempt's MXFP8 path plus INKFASTPRE-03): pooled over 2,580 rows, next-token accuracy 73.41 percent against the exact reference's 73.64, 99.28 percent confident top-1 agreement; passes the quality bar ([INKROUND-08](2026-11-23-inkling-mini-decode-round.md#20261123-inkround-08--natural-text-quality-probe-and-the-synthetic-capture-finding)). The MXFP8 change alone was not measured on natural text. Committed in `2a1d72a2`.
