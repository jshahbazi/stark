# 20261122-INKFUSE-01 · Inkling 2 Mini verifier and single-row launch fusion, first pass

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKFUSE-01` / 2026-11-22 / 2026-11-22 |
| Kind | Kernel (launch fusion) with correctness checks |
| Attempt state | Completed for the first pass; the follow-up fusions listed below are not started |
| Result and acceptance criterion | Passed: exact pinned 4K parity and an unchanged transaction report. Timing inconclusive: no paired measurement. |
| Evidence scope | Full-model numerical (one pinned stock 4K capture), transaction self-consistency on one 4K capture, dispatch counts. One unpaired benchmark invocation per arm. |
| Promotion decision | Adopted in the tree; speed unqualified |
| Disposition | Adopted on branch, uncommitted on top of `f1666090`. Not merged, released or installed. |
| Predecessor / related attempts / PR | Builds on [20261122-INKDEFAULT-01](2026-11-22-inkling-mini-default-fast-paths.md). Long checks passed later on the [20261122-INKLONGPOS-01](2026-11-22-inkling-mini-long-position-gain.md) fix build. First-look timing in [20261122-INKENGINE-01](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md). No PR. |

## Change

Each fused kernel keeps the replaced chain's operation order, BF16 rounding points, reductions and pragmas.

- Verifier residual mix plus the q, k and v norms in one launch (`ink_verify_mix_qkv_norm`), and expert mix plus the MLP norm in one (`ink_verify_mix_expert_norm`). The batched first-pass norm now covers two rows as well.
- Per-visit row preparation in one launch (`ink_verify_prepare_rows`).
- Post-attention convolution, residual, boundary copy, expert mix and first-pass MLP norm in one launch (`ink_verify_attention_mix`), with its convolution history writes deferred to `ink_verify_finish_experts_history`.
- The single-row source path runs through the same kernels with one row (`Session::encode_fused_visit`). Its later-pass controller chain falls from 17 launches to 5, the expert sum, MLP convolution and residual run in one launch, and it uses the native M16 × N8 router.

Source: [`verify_first_norm.metal`](../../crates/stark-engine/src/engine/inkling_slice/verify_first_norm.metal), [`verify_state.metal`](../../crates/stark-engine/src/engine/inkling_slice/verify_state.metal), [`verify_finish.metal`](../../crates/stark-engine/src/engine/inkling_slice/verify_finish.metal), [`verify_state.rs`](../../crates/stark-engine/src/engine/inkling_slice/verify_state.rs), [`prefill.rs`](../../crates/stark-engine/src/engine/inkling_slice/prefill.rs), [`grouped_pilot.rs`](../../crates/stark-engine/src/engine/inkling_slice/grouped_pilot.rs).

## Artifact, source and environment

| Field | Value |
| --- | --- |
| Checkpoint | Original `thinkingmachines/Inkling-2-Mini` at `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, hashes as in the [default-router record](2026-11-21-inkling-mini-decode-default.md#artifact-and-run-controls) |
| Baseline build | Defaults only, `stark-generate` SHA-256 beginning `2da8222d57cf` |
| Candidate build | Defaults plus this fusion pass, before the long-position fix: `stark-generate` beginning `133f76c237df`. The matching `inkling_slice` example hash is not recorded. |
| Source | Uncommitted working tree on top of `f1666090`; no patch hash recorded |
| Host | 128 GiB Apple M5 Max (Mac17,7), macOS 26.7.2 (25G332), AC power, `pmset` powermode 2 |

## Launch counts

Launches per token from the engine's dispatch counter on the `g4096-02` capture:

| Path | Before | After | Change |
| --- | ---: | ---: | ---: |
| Single row, including the draft commit | 1,907 | 1,368 | −539 |
| Two-row verifier | 1,670 | 1,417 | −253 |
| Three-row verifier | 1,541 | 1,422 | −119 |
| Four-row verifier | 1,604 | 1,475 | −129 |

## Correctness

- `--full-target-batched` on `s4096-00` was exact against the pinned capture on all four rows. Every logits, taps, hidden, partial-state and history hash was identical to the pre-fusion build.
- `--measure-verify-transaction` on `g4096-02` was identical to the pre-fusion build in every non-timing field.
- Known side effect: the parity report's `attention` and `trace.mlp_conv` trace probes now read stale scratch buffers and report every element unequal, 24 fields each per capture. Logits and hidden state are unaffected. Those two probes cannot serve as evidence on this build until they are fixed.
- Not checked above 4K at the time. The later fix build ([20261122-INKLONGPOS-01](2026-11-22-inkling-mini-long-position-gain.md)) passed the long transaction-schedule checks with this fusion in place.

## Performance

No paired measurement was taken. In the first-look engine matrix ([20261122-INKENGINE-01](../benchmarks/2026-11-22-inkling-mini-engine-matrix.md)), one invocation per arm, the fused build was about 2 percent ahead of the defaults-only build on most rows and behind on novel prose:

| Scenario | Defaults only | With fusion |
| --- | ---: | ---: |
| Initial coding, n = 3 | 76.3 tok/s | 78.0 tok/s |
| Sustained coding, n = 3 | 77.1 tok/s | 77.0 tok/s |
| Novel prose, n = 3 | 65.3 tok/s | 58.6 tok/s |
| Context about 4K, n = 1 | 104.1 tok/s | 106.5 tok/s |

Tokens per step were equal between the two Stark arms in each short scenario. The prose regression is unexplained; it may be noise in a single invocation or a real cost, and this run cannot tell which. The speed effect of this pass is unqualified.

## Decision and next steps

Adopted in the tree with speed unqualified and the prose result unexplained. A paired same-host comparison of the two builds is required before any speed claim. Not started: folding down-activation quantization into the down kernel, cross-visit fusion, the remaining single-row K/V and controller launches, and removing the `STARK_INKLING_SOURCE_PREFILL` kill switch. Performance playbook: not updated with this record; any update is a separate change.

Keep prompts, completions, token vectors and probabilities out of this document; the private reports stay ignored.

## Follow-up history

### 20261122-INKFUSE-02 · Second session: paired timing and single-row fusions

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date | `20261122-INKFUSE-02` / 2026-11-22 / 2026-11-22 to 2026-11-23 |
| Kind | Kernel (launch fusion) with paired timing and correctness checks; probe repair; kill-switch removal |
| Result | Passed: exact pinned 4K parity and transaction checks. Paired timing: single-row step 15.6% faster; verifier commands flat. The INKFUSE-01 prose regression did not reproduce. |
| Evidence scope | Paired engine-diagnostic timing (six pairs on the final build), full-model numerical on two pinned 4K captures, transaction self-consistency to 16,000 tokens, dispatch counts |
| Promotion decision | Adopted in the tree. The single-row gain is measured on the engine diagnostic only; the worker effect is not measured. |
| Disposition | Adopted on branch, uncommitted on top of `f1666090`. Not merged, released or installed. |

**Changes this session**

- Single-row source path (now the default): the six attention projections, plus the two dynamic-convolution projections on global layers, run in one launch (`ink_matvec_source_visit`) instead of six to eight, with the unchanged single-row dot and the same K-lane count per output segment. The K/V convolution, normalization and cache write and the Q/R normalization run in one launch (`ink_verify_heads_pack_one`) instead of three, using the verifier's pack bodies unchanged. The later-pass MLP controller uses the verifier's native rank and expansion chain with one row: 4 launches instead of 5.
- Trace probes repaired: `attention`, `trace.mlp_conv`, `trace.key_conv` and `trace.value_conv` report zero unequal in every capture. Only when visit capture is on, two small kernels (`ink_verify_conv_probe`, `ink_verify_kv_conv_probe`) recompute those outputs from the history before the slot is overwritten; production does not run them.
- `STARK_INKLING_SOURCE_PREFILL` is removed from `packed_prefill.rs` and `Session::batched_prefill_memory_bytes`. The source path no longer has an environment kill switch.

Source, in `crates/stark-engine/src/engine/inkling_slice/`: `mod.rs`, `prefill.rs`, `prefill_controller.rs`, `session.rs`, `packed_prefill.rs`, `serial.metal`, `verify_cache.metal` and `verify_finish.metal`.

**Launch counts** (engine dispatch counter, `g4096-02`): single row including 24 draft-commit launches 1,907 before fusion, 1,368 after INKFUSE-01, 992 now. Verifier unchanged since INKFUSE-01: 1,417 / 1,422 / 1,475 at two, three and four rows.

**Paired timing.** Alternating base, candidate, candidate, base against the pre-fusion defaults build (`inkling_slice` `b45cbce0d2dbff7001029c7da260e27823d0ad3185974de80f3bac9a5f92876d`). Each run did `--measure-verify-transaction` on `g4096-02` (96 tokens: serial decode, then depths 1 to 3 with the real draft) plus the profiling mode for the two-, three- and four-row verify commands. The candidate ran with `--exact-prefill` so its prefill matched the base's exact arithmetic. The host was shared with another agent's GPU jobs, serialized through the lock. Final build, three cycles (six pairs), median of per-pair ratios:

| Path | Pre-fusion | Fused | Ratio | Notes |
| --- | ---: | ---: | ---: | --- |
| Single-row step | 22.86 ms | 19.36 ms | **0.844** | Fused faster in 6 of 6 pairs |
| Two-row verify command, GPU | 20.54 ms | 20.33 ms | 0.989 | |
| Three-row verify command, GPU | 22.52 ms | 22.55 ms | 1.002 | |
| Four-row verify command, GPU | 24.98 ms | 24.52 ms | 1.000 | |
| Depth 1 decode | | | 1.010 | 73% acceptance |
| Depth 2 decode | | | 1.016 | 54% acceptance |
| Depth 3 decode | | | 0.994 | 40% acceptance, the most rejections |
| Commit per round, all depths | | | 0.96–1.01 | |

An earlier four-pair run on a mid-session build (verifier fusions plus the first single-row fusions) gave single row 0.897 and verifier commands 0.99 to 1.00. A third run compared the pre-fusion build with this author's build from before other agents' concurrent changes and found the same per-width verifier GPU time, which places the flat verifier result on these kernels.

Rejection-heavy rounds (depth 3, 40% acceptance) show no regression: the commit path (undo copies plus draft commit) runs the same code as before, and the deferred history writes happen inside the verify command.

**The INKFUSE-01 novel-prose result (58.6 against 65.3 tok/s) did not reproduce.** In the first pair of the first run the fused build was 5 to 17% slower on every stage, including attention projections and experts that the fusion does not touch; every later pair was level. That pattern points to host state, not code. No fix was made.

**Where verifier round cost goes.** In a four-row profile, experts take 11.8 ms, attention projections 5.8 ms and attention 2.8 ms of 27 ms GPU time. Removing 120 to 250 launches per round changed nothing measurable, so launch count is not the verifier's cost driver on this host.

**Correctness, final build** (exact-prefill runs):

- `--full-target-batched` on `s4096-00` and `g4096-01-retry` at 4,352: logits exact on all four rows, snapshot replay exact, and all logits, taps, hidden and state hashes identical to the INKFASTPRE-01 build. One probe field, `down_quantized`, is unequal by 2 in `g4096-01`, as in the baseline.
- `--measure-verify-transaction` on `g4096-02`: zero non-timing differences from the INKFASTPRE-01 build.
- `--measure-verify-transaction-schedule` at s73/4,352, r4150/4,352, l8800/9,216 and l16000/16,384: pass, including rejection and crossing controls, exact boundary replays and serial match at depths 1 to 3. The only difference from the INKFASTPRE-01 build is the new `exact_prefill_reference` field.
- Comparisons against the INKFASTPRE-01 build show 144 probe fields as differences because that build still has the stale probes.

**Identity.** The final binary is not bound by hash in the report. The final checks are in the author's `E3` output directory, and the scratch binary named `inkling_slice_E3` has SHA-256 `64ec58b17dc09ab1f41d67391f5e91203c1f138bd6b2b7638dd4c73e1e6f26b5`; treat that binding as probable, not confirmed.

**Not done, with reasons**

| Item | Reason |
| --- | --- |
| Cross-visit fusion (expert finish into the next visit's mix and norm) | Race-free only where the next kernel has one workgroup per row (about 18 local first-pass visits per round). Global and later-pass visits use several workgroups per row. With verifier launches measured as free, the expected gain is about zero. |
| Folding down-activation quantization into the down kernel | Each down workgroup would repeat the full quantization of 1,152 values per route, about 576-fold redundant arithmetic in the hottest kernel. Ceiling about 44 launches per token, roughly 0.2 ms on the single row. |
| Verifier down epilogue into the expert finish | A clean exact fusion candidate, but it needs an edit in a file another agent owned, for about 44 launches per round worth about nothing on current evidence. |
