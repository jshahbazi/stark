# 20261119-LILYROWS-01: shared-weight hyper-connection reads in MTP verify

Registered and executed 2026-11-19. Owner: Claude (workstream `hc-rows`).
State: completed; not adopted (see Follow-up history).

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | 20261119-LILYROWS-01 / 2026-11-19 / 2026-11-19 / Claude, workstream `hc-rows` |
| Kind | Kernel scheduling (MTP verify hyper-connection reads) |
| Attempt state | Completed |
| Result and acceptance criterion | Partial. Shared-weight exact down half: passed byte tests and the end-to-end hash gate. Shared-weight exact up half: failed (not byte-exact under fast math) and rejected. Target HC verify time of about 3.2/3.4/3.6 ms at 2/3/4 rows: not reached |
| Evidence scope | Operator/component byte tests, synthetic-weight component timing, full-model end-to-end output hashes and timing (screening only) |
| Promotion decision | Not adopted. Down half: byte-exact, but its gain is below measurement resolution, so it was not merged on risk against benefit (the GDN follow-up later showed synthetic byte tests can pass while the real model differs). Up half rejected |
| Disposition | The code is not on the integration working tree. The source patch `hc-rows.patch` is archived as unadopted experimental source in [`docs/archive/experiments/2026-11-19-lily-verify-rows/`](../archive/experiments/2026-11-19-lily-verify-rows/README.md); not committed, released or installed |
| Predecessor / related attempts / governing issue / PR | 20261119-SUSHI-02 (baseline profile), 20261116-LILY-REUSE-01 (projection reuse precedent, commit `63c7601f`); no PR |

## Hypothesis and change

- Problem: every MTP verify hyper-connection (HC) read runs the one-row
  down and up bodies once per row on the grid's row axis
  (`stark_hc_down_decode_rows`, `stark_hc_up_cached_rows_g<G>`). The
  baseline stage profile shows HC verify GPU time of 2.91 ms (serial step),
  4.26, 5.39 and 6.37 ms at 1 to 4 rows, about 1.15 ms per extra row.
- Hypothesis: the per-row cost is repeated weight traffic, so reading each
  weight word once for all rows (as projection reuse does) would flatten it.
- Candidate: `hc_read_verify_exact` (Lily, non-official rounding path) now
  encodes the down half with the existing shared-weight kernel
  `hc_read_down_q8_m<rows>` (`hc_read_down_q8_rows`), which unpacks each Q8
  word and loads each scale and bias once and keeps an independent
  per-row accumulation with the MB=1 lane order. The up half keeps the
  row-axis one-row cached body.
- Fixed criteria set before the runs: byte equality of every output field
  and row against the MB=1 bodies for all supported row counts, real
  dimensions, edge geometries, zeros, subnormals and cancellation, offset
  views and output canaries; then the real-model verify/rollback test and
  the baseline output-hash gate for all eight arms.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at `0f94842e3682b49acace4daeb4cf2e65a6f94f83` | Same |
| Format/quantization | Lily Q4-XL; HC down/up/inject weights Q8 group 64 | Same |
| Backend, vendor revision and local patch provenance | `vendor/lily-core` at base `4d09b550` | Base plus this workstream's HC call-site change |
| Stark commit, dirty patch and executable hash | `4d09b550`; worker `5dfb90b6...298d` | `4d09b550` plus `runtime/patches/hc-rows.patch`; worker `d65919028cdb050fec4a8c8a47e303c712ac5910617872d3dabd9bf0d06363f8` |
| Toolchains and build flags | Rust 1.98.1 release, `--locked` | Same; tests with `nightly-2026-10-01` |
| Weight, activation, accumulation, recurrent-state and KV precision | Mixed Q4/Q8 weights, BF16 activations, F32 accumulation | Unchanged |
| Selected kernel and materialization policies | Verify HC down: row-axis MB=1 body | Verify HC down: shared-weight `hc_read_down_q8_m<rows>`; up unchanged |
| Hardware, OS, Metal toolchain | Apple M5 Max, 128 GB, Darwin 25.6.0, Metal front end 32023.883 | Same |
| Memory | Not measured | No new buffers |

## Workload and controls

- Image-loaded default worker, 4,096-token frozen public code corpus, fresh
  requests, arms g0 to g3 (greedy, 512 tokens) and s0 to s3 (temperature 1,
  top-k 20, top-p 0.95, seed 42, 256 tokens), one repeat, host shared with
  other agents (screening only), via the shared model-slot wrapper.
- Component timing: `verifier_rows_streaming_timing` (ignored test in
  `hc_tests.rs`), 97 distinct synthetic Q8 weight sets at H=2560, G=4,
  R=320, group 64 (about 680 MB, so weights stream from DRAM), down and up
  halves timed in separate chains, 12 measured passes per configuration in
  rotating order after a 2 s warmup. The GPU was shared with other agents.
- Prefix reuse and image paths: not affected (kernel substitution inside
  the verify graph only); not separately exercised beyond the hash gate.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| A, cost location | Component timing of the existing row-axis kernels against shared-weight kernels | Row-axis down+up per pass of 97 reads: about 1.86, 2.38-2.49, 3.09-3.29, 3.83-4.03 ms at 1-4 rows. Even the non-exact prefill shared kernels still grow about 0.2-0.3 ms per row in the down half. Weight traffic is not the main per-row cost | Valid screening; see reading below |
| B, down half | Existing `hc_read_down_q8_m{2,3,4}` as the verify down kernel | Byte-identical to the MB=1 body in all 1,080 cases (3 draws x 5 geometries x 4 row counts x 6 styles, with and without inject, plus up-only cases). About 0.06, 0.15-0.2, 0.2-0.25 ms faster per pass than the row-axis down at 2, 3, 4 rows | Adopted (pending orchestration) |
| C, up half, register template | New `stark_hc_up_cached_shared_g<G>_m<MB>`: weights, scales and biases once per lane, per-row accumulation with the cached body's expression | IR identical per row to the one-row body, yet rows 1-3 differed in up to 300 of 2,560 bf16 outputs with subnormal streams and in single outputs on ordinary random inputs | Not exact; rejected |
| C2 | Same with a rolled (unroll-disabled) per-row epilogue | Passed one draw of all styles; failed a second draw with one differing bf16 output (style random, rows 2, row 0) | Not exact; rejected |
| C3 | Same with epilogue inputs parked in threadgroup memory | Failed G=1 subnormal cases | Not exact; rejected |
| D, loop kernels | One threadgroup/simdgroup running the unchanged MB=1 down or cached up body for every row in turn (cache-resident weights) | Byte-exact, but slower than the row-axis kernels at every row count (serialized latency) | Rejected on speed |

Diagnosis of the up failures: the AIR for the template matches the one-row
body per row; the differences come from the GPU back-end's fast-math
scheduling of the mix epilogue (`sigmoid(logit) * bf16(hyper * inv_rms *
gain)` accumulation), which changes with the surrounding multi-row code.
Inputs that keep the normalized streams normal were exact; subnormal
streams and rare rounding boundaries were not. Source-level equivalence is
therefore not sufficient for a shared-weight up kernel at this precision
policy, and an empirically passing variant was found to be draw-dependent.

## Correctness and lifecycle

- Kernel byte tests (physical Metal, `nightly-2026-10-01`, release):
  `verifier_shared_rows_match_ordinary_hc_bytes` (new) 1,080 cases,
  4,620,960 compared bytes, 0 unequal, reference and candidate SHA-256
  `5f0650be90410c2eb6f9b7d48f3d4b6d68814de42b00e1897d3a45573b6a49fe`;
  `verifier_cached_rows_match_ordinary_hc_bytes` 128 cases, 699,360 bytes,
  0 unequal; `cached_read_is_bit_identical_to_single_row` 28 cases, equal
  hashes.
- Real-model exactness test `private_checkpoint_mtp_verify_and_rollback_match_serial`:
  passed (1 test, 54.7 s, model under the shared slot, `STARK_TEST_MODEL_PATH`;
  the brief's `STARK_LILY_TEST_MODEL_PATH` name panics on the missing fixture
  variable, which was a first invalid attempt).
- End-to-end gate `runtime/agent-runs/hc-rows-1.json`: input hash matches
  the baseline; output hashes and MTP work counts equal the baseline for all
  eight arms.
- No scratch, buffer or lifecycle change: the shared kernel writes the same
  `[rows, ...]` outputs into the existing verify scratch.

## Performance and memory

Component timing (median GPU ms per pass of 97 HC reads, run 3 of 4, shared
GPU; other runs agree within about 10 percent):

| Rows | Row-axis down | Shared down (adopted) | Row-axis up (kept) | Shared up, not exact |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.974 | 0.963 | 0.879 | 0.908 |
| 2 | 1.185 | 1.087 | 1.302 | 0.929 |
| 3 | 1.571 | 1.363 | 1.716 | 1.052 |
| 4 | 1.899 | 1.681 | 2.134 | 1.523 |

End-to-end screening (one repeat, shared host, uninstrumented): tok/s
change against the baseline g0 +3.41, g1 +3.85, g2 +1.69, g3 +1.08, s0
-2.32, s1 -2.54, s2 +0.23, s3 +3.69 percent. The MTP-off arms (g0, s0), which
this change does not touch, moved by the same magnitude, so these are noise
and no end-to-end speed claim is made.

Stage profile (`qwen.mtp.verify.gpu.hc`, ms per verify pass):
Two instrumented sessions on a contended host. In both, the untouched
serial-step HC and GPU command totals were 10 to 20 percent above the
baseline session, so absolute values are not comparable.

| Session | Serial HC (s0) | 2 rows (s1) | 3 rows (s2) | 4 rows (s3) | Verify command at 2/3/4 rows |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline 20261119-SUSHI-02 | 2.907 | 4.257 | 5.389 | 6.374 | 19.52 / 25.29 / 30.96 |
| `hc-rows-2` (s1-s3; warmup s0 only) | 3.295 (warmup) | 4.434 | 6.407 | 7.651 | 20.92 / 27.09 / 32.70 |
| `hc-rows-3` (s0-s3) | 3.477 | 4.826 | 4.943 | 5.955 | 20.76 / 25.19 / 32.50 |

The two candidate sessions disagree by up to 30 percent at 3 rows, so the
in-model HC effect is inconclusive at this noise level. Run files:
`runtime/agent-runs/hc-rows-2-staged.json`, `hc-rows-2-stages.jsonl`,
`hc-rows-3-staged.json`, `hc-rows-3-stages.jsonl`.

## Evidence, decision and limits

- Reading: the per-row HC cost is mostly not weight traffic. In the down
  half each of the 324 output-row threadgroups rereads the full 20 KB hyper
  row and recomputes its squared norm, about 6.6 MB of cache traffic per
  row per read, twice the weight bytes; in the up half each extra row adds
  per-column unpack, dot and a four-way SIMD reduction plus the mix
  epilogue. Sharing weight loads removes only part of this, and the up half
  cannot share them exactly under the present fast-math compilation.
- Adopted within scope: shared-weight down for Lily verify rows 2 to 4 (and
  1). Rejected: every tested shared-weight up variant.
- Row counts beyond 4 (to 9): the down kernel is a template on MB; adding
  instantiations and raising `HC_FUSED_MAX_ROWS` would need its byte test
  rerun at those counts. Not done.
- Not verified: quiet-host timing; an in-model HC improvement (the stage sessions were
  too noisy; the orchestrator's quiet A/B is needed).
- Next action: an exact flattening of HC rows would need either a
  multi-output-row down threadgroup that stages the normalized row once
  (also a candidate for the one-row decode HC, about 20 percent of a serial
  step), or an explicitly ordered (non-fast-math) one-row HC body adopted
  for decode and verify together, which changes decode numerics and needs
  its own precision qualification.

## Local patch provenance (vendor/lily-core)

- `src/qwen4exp/model.rs`: `hc_read_verify_exact` calls
  `hc_read_down_q8_rows` (kernel `hc_read_down_q8_m<rows>`) instead of
  `hc_read_down_q8_decode_rows`; import list adjusted.
- `src/kernels/hc.rs`: documentation of the exactness reliance only.
- `src/kernels/hc_tests.rs`: new byte test
  `verifier_shared_rows_match_ordinary_hc_bytes` and diagnostic timing test
  `verifier_rows_streaming_timing`; `GuardedTensor::rows` and
  `ReadOutputs::rows` helpers.
- No Metal source change. Sushi source was not read or adapted.

## Follow-up history

- 2026-11-19, integration: the down half was not adopted. Its byte-exact gain
  is below measurement resolution, so it was judged not worth its risk; the
  GDN follow-up of 20261119-LILYROWS-02 later found a kernel that was
  byte-identical in synthetic tests yet changed real-model verify values. The
  code described above, including the "adopted within scope" down kernel and
  the local patch provenance, is not on the integration working tree. The
  source patch `hc-rows.patch` (SHA-256
  `8e39514c3ed1c7e4652422bcd342ccc80f2a24fd2d0ac92914ca20e70ad06308`) is
  archived unapplied in
  [`docs/archive/experiments/2026-11-19-lily-verify-rows/`](../archive/experiments/2026-11-19-lily-verify-rows/README.md).
