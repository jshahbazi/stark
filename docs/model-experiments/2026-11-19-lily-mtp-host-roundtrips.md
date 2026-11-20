# 20261119-LILYHOST-01: fewer host round trips in Lily MTP rounds

Registered and executed 2026-11-19. Owner: host-roundtrip workstream of the
Lily MTP speed project. State: attempted; awaiting the orchestrator's quiet
comparison. Kind: sampling and scheduling. Disposition: merged into one
uncommitted integration working tree (see [Integration](#integration)); not
committed, installed or released; final quiet timing pending.

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | 20261119-LILYHOST-01 / 2026-11-19 / 2026-11-19 / host-roundtrip workstream |
| Kind | Sampling and scheduling (host-side work in MTP rounds) |
| Attempt state | Attempted |
| Result and acceptance criterion | Output-hash gate passed on all eight arms for every candidate build, including the final source. Screening timing (not quiet-screened): s2 +8.6%, g2 +12.9% against a contemporaneous base rebuild. Criterion: every baseline output hash reproduced; sampler tests unchanged and passing |
| Evidence scope | Operator/component (host sampler micro-timing, Metal argmax control), end-to-end output hashes and screening timing |
| Promotion decision | Adopted on the integration working tree; the private real-model control passed there. Production promotion awaits the final quiet comparison |
| Disposition | Merged with 20261119-LILYROWS-02, 20261119-LILYDEPTH-01 and 20261119-LILYBENCH-01 into one uncommitted working tree; not committed, not installed, final quiet timing pending |
| Predecessor / related attempts / governing issue / PR | Baseline 20261119-SUSHI-02; no PR |

## Hypothesis and change

- Problem: an MTP round is fully synchronous. Each draft step, the verify pass
  and the finish pass commit and wait, and the host copies, scans and samples
  complete F32 vocabulary rows (248,320 values) between them. The baseline
  measured about 2.5 ms of host-side cost per two-draft round sampled and
  3.8 ms greedy, with one greedy sampler call at 0.357 ms against 0.086 ms for
  top-k 20.
- Changes, all scheduling or host-side:
  1. Greedy sampler. The former scan `if logits[i] > logits[best]` reloaded
     `logits[best]` every iteration, a loop-carried dependency that does not
     vectorize, after a separate finite pass. `greedy_index` computes 32 lane
     maxima and finiteness in one vectorizable pass, then finds the first
     element equal (`==`) to the maximum. On finite input this is the same ID:
     the scan settles on the first member of the maximal class under `==`
     (signed zeros equal). Nonfinite input still fails.
  2. Encode ahead. Draft step `i + 1`, and then the verify graph, are encoded
     while the preceding draft pass runs. Neither encoding reads a token value;
     ids and n-gram rows are written to scratch after the predecessor completes
     and before commit, the pattern `GreedyDecode` already uses. State and
     scratch growth for the verify batch happen before the first draft encode,
     and committing the early verify encoding checks that position, convolution
     slot, state capacity and scratch identity did not change.
  3. Greedy device draft chain. For a sampler that reports
     `device_greedy_drafts` (ExactMtpSampler with greedy sampling), all draft
     steps run in one command buffer: the same step graphs, each step's
     proposal the existing device first-max argmax (`argmax_f32`) feeding the
     next step's embedding gather. Only the proposal ids and one status word per
     step are read back. The proposals are recorded with the new trait method
     `draft_point_mass`, which stores the same one-candidate distribution that
     `draft` stores for a greedy row. Verify readout is unchanged (complete F32
     rows on the host), so target selection and acceptance stay on the host.
  4. Redundant per-row work. Draft and verify rows are borrowed from shared
     storage instead of copied (`to_f32`, 993 KB per row). Verify rows are
     validated as they are consumed; rows after the first rejection are neither
     copied nor scanned.
  5. Polled waits. Draft steps, the greedy chain and the verify pass wait with
     `PendingPass::wait_polled` (the done-event poll `GreedyDecode` already
     uses) instead of the blocking fence wait. Completion is still confirmed
     through the fence and feedback errors are still checked. The finish pass
     (`finish_host_verify`) is outside this workstream and still blocks.
- Device argmax exactness. A Metal control found that device comparisons flush
  subnormals: with a subnormal maximum among zeros the device chose index 0
  where the host chose a later subnormal. The chain therefore marks a step
  whose selected logit is a zero or subnormal (`stark_flag_tiny_selection_f32`).
  A selected normal value is exact: flushed operands are all below it and
  comparisons between normals are exact. On a flagged step the round discards
  the chain and reruns the host path from the same state; the chain wrote only
  head-cache rows at this round's speculative positions, which the rerun
  rewrites before reading, and scratch. A nonfinite readout sets the status to
  `u32::MAX`; the host fails the round only if the first nonzero status word is
  nonfinite, the step at which the host path would fail. The device id written
  for the next step is always in range, so no sentinel reaches a gather.
- Not changed: target logits, verify graph arithmetic, rollback
  (`finish_host_verify`, `encode_draft`), sampled-request draft sampling, RNG
  streams and draw order. No candidate cap was introduced.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, image tower loaded | same |
| Source | `4d09b55045f24eab81da0a9b5b024b9c02318459` | same base plus the uncommitted patch `runtime/patches/host-roundtrip.patch` |
| Worker SHA-256 | Reference baseline `5dfb90b6...298d`; contemporaneous rebuild of the base source in a scratch path `de6b2c9c1ebbfa558abf74d2f9dccf30bc705d1abf6dd2405eb1357d26a401a0` | v1 (items 1, 2, 4) `c193d49d519e96d3905698553904425e929e0dd37244becc0111ccebfb523252`; v3 (items 1 to 4) `de781b5b05c8989542cca2da4d7be98693daf0e57615fde977ef74791579552c`; v5 (adds polled waits) `06842f7908f1dd13515a5b180827790362553f256eb5e07d4bb69af022e19cb6`; final source `acd776de39bfdcffeda066dbc8a09dc5e72fbb29c87b4891f94be5ba71301ddb` |
| Toolchain | Rust 1.98.1, locked release build | same |
| Precision | Unchanged: Q4/Q8 weights, BF16 activations, F32 accumulation, readouts and GDN state, Q8 KV | same |
| Host | Apple M5 Max, 128 GB, macOS 26.7.2; other workstreams compiling and running GPU tests concurrently | same |

## Workload and controls

- Runner: `runtime/sushi-profile-2026-11-19/run_profile.py` with the frozen
  corpus, 4,096 input tokens, seed 42; sampled arms temperature 1, top-k 20,
  top-p 0.95, 256 tokens; greedy arms 512 tokens; image tower loaded; every
  run under the model-slot wrapper.
- Hash gate: `compare_to_baseline.py` against attempt 20261119-SUSHI-02.

## Attempt history and failures

| Trial | What was attempted | Observation | Disposition |
| --- | --- | --- | --- |
| Greedy micro-timing (synthetic 248,320 values) | Old scan versus `greedy_index` | Scan 0.273 ms (plus 0.014 ms finite pass); new 0.030 ms including finiteness; top-k 20 call 0.094 ms | Adopted |
| Device argmax control, first version | Device `argmax_f32` versus host first-max on ties, signed zeros, subnormals | Mismatch on a subnormal maximum (device index 0, host index 17 at 255 values) | Fixed by the tiny-selection flag and host rerun |
| `host-roundtrip-1` (v1) | Full eight-arm gate | All hashes equal; timing perturbed by a concurrent build | Hash evidence only |
| `host-roundtrip-2` (v2: chain without tiny flag) | Full gate | All hashes equal | Superseded by v3 (inexact on subnormal maxima) |
| `host-roundtrip-3` (v3), `host-roundtrip-4` (base rebuild) | Full gate, sequential pair | All hashes equal for both; base rebuild ran 5 to 30% below the reference baseline, so the host was loaded | Screening only |
| `host-roundtrip-13` (v5), `host-roundtrip-23` (final source) | Full eight-arm gate | All hashes equal | Hash gate passed for the final source |
| Formatting slip | `rustfmt` on `qwen4exp/mod.rs` recursed into `request_memory.rs`, `vision.rs` and `vision_reference_tests.rs` | Formatting-only changes outside scope | Reverted to the base revision; not part of the patch |
| Considered, not done | Device argmax of verify rows; GPU top-k readback for sampled requests | Host argmax of a borrowed verify row now costs about 0.04 ms; a separate device pass would add a commit and wait. Top-k readback deferred | Not attempted |

## Correctness and lifecycle

- Unchanged sampler tests in `crates/stark-generate/src/hybrid/mtp.rs` pass.
  Worker library tests: 147 passed, 5 ignored. Core library tests: 97 passed,
  58 ignored. The engine test target compiles with `nightly-2026-10-01`; the
  full engine suite was not run.
- New tests: `greedy_index_matches_scalar_scan_on_ties_signed_zero_and_nonfinite_inputs`
  (worker; lengths 0 to 248,320, ties in every lane position and the tail,
  signed-zero classes, five nonfinite encodings at three positions);
  `greedy_point_mass_drafts_match_host_greedy_drafts` (worker; proposals,
  outputs and both RNG states equal for depths 1 to 3 and every rejection
  point, plus fail-closed misuse); `chained_argmax_matches_host_first_max_and_flags_nonfinite_in_range`
  (core, physical Metal, passed: exact whenever the status is clear, flagged
  exactly when the host maximum is a zero or subnormal, nonfinite flagged with
  an in-range id).
- Private real-model control `private_checkpoint_mtp_verify_and_rollback_match_serial`
  gains a device-versus-host greedy section (depths 1 to 3, early drops after
  0, 1, 2, 5 and 8 tokens, equal outputs, positions and byte-equal next-step
  logits). Run on the final source under the model slot with
  `STARK_TEST_MODEL_PATH`: passed (1 test, 53 s). Verify, rollback and
  host-serial phases have 0 unequal values; the head phase differs as before
  (draft-head numerics may differ by contract).
- Cancellation, early EOS and `Drop`: no pass is left submitted when `round`
  returns. An encoded but uncommitted pass (next draft step or verify) is
  dropped without submission on any error; a committed pass dropped on error
  waits for the GPU in `PendingPass::drop`. `Drop` for `MtpDecode` and its
  rollback are unchanged and still see only synchronous state.
- Validation change for rows never consumed: previously a nonfinite value in a
  verify row after the first rejection failed the round. Now such a row is not
  read. Its values cannot affect retained state: the rollback keeps state only
  through the rejected row, which is validated, and later rows' cache entries
  lie beyond the new position. Every row handed to the sampler is still
  validated by the engine, and the sampler validates again; the engine check is
  kept because the `MtpSampler` trait does not require implementations to
  validate.

## Performance and memory

Screening on a loaded host. Draft phase per round (ms), from the full-gate runs:

| Arm | Reference baseline | Base rebuild (run 4) | v3 (run 3) |
| --- | ---: | ---: | ---: |
| g1 / g2 / g3 | 1.80 / 3.54 / 5.26 | 2.64 / 6.03 / 9.32 | 1.34 / 2.44 / 3.57 |

Interleaved A/B on the same host (arms s2, g2, s3, g3, two repeats per run,
medians of four requests per arm). Base rebuild runs 14 and 20; v4 (items 1
to 4) runs 15 and 21; v5 (items 1 to 5, functionally the final source) runs
16 and 22. Runs 17 to 19 were cut off by a pause and are discarded.

| Arm | Base tok/s (round ms) | v4 tok/s (round ms) | v5 tok/s (round ms) | v5 draft / verify / finish ms | Base draft / verify / finish ms |
| --- | --- | --- | --- | --- | --- |
| s2 | 78.46 (29.13) | 79.07 (28.91) | 85.21 (26.83) | 2.66 / 23.02 / 0.90 | 2.93 / 25.10 / 0.90 |
| s3 | 68.16 (36.47) | 66.12 (37.60) | 73.24 (33.94) | 3.98 / 28.64 / 1.10 | 4.38 / 30.72 / 1.08 |
| g2 | 91.14 (30.53) | 92.06 (30.26) | 102.91 (27.04) | 2.30 / 23.74 / 0.90 | 3.49 / 25.11 / 0.92 |
| g3 | 89.53 (38.64) | 95.59 (36.21) | 94.89 (36.49) | 3.71 / 31.36 / 1.21 | 5.18 / 30.52 / 1.61 |

Earlier interleaved series (runs 5/7/9 base against 6/8/10 v3, six requests
per arm): g2 90.18 to 95.72, g3 88.04 to 95.73, s2 77.00 to 78.34, s3 66.83
to 67.89 tok/s.

Instrumented pair (runs 11 base and 12 v4, s2 and g2; instrumentation
perturbs timing, attribution only):

| Counter (ms per round unless noted) | Base s2 | v4 s2 | Base g2 | v4 g2 |
| --- | ---: | ---: | ---: | ---: |
| `qwen.mtp.draft` wall / GPU busy | 2.90 / 2.08 | 2.85 / 2.10 | 3.50 / 2.13 | 2.41 / 2.09 |
| `qwen.mtp.verify` wall / GPU busy | 27.36 / 26.20 | 27.09 / 26.62 | 27.57 / 26.35 | 27.02 / 26.59 |
| `qwen.mtp.target_sampling` | 0.19 | 0.25 | 1.13 | 0.10 |
| Greedy sampler call (`sample.total`) | | | 0.360 | 0.023 |

Reading: the greedy sampler and device chain remove most greedy host time,
as predicted. Encode-ahead and borrowed rows alone were worth only about
0.1 to 0.3 ms per sampled round: draft-step encoding is small, and most of
the sampled draft gap is submission and wake-up latency plus the sampler.
Polled waits gave the largest sampled gain, about 2 ms of verify wall per
round, more than the documented 0.09 ms fence wake-up latency per wait. The
cause of the excess was not isolated; a plausible explanation is that the
host thread does not sleep through the 25 ms verify and so avoids a slow
wake, but this was not measured. The final full gate (run 23, worker
`acd776de...1ddb`) also ran faster than the reference baseline on the MTP-off
arms (s0 +3.4%, g0 +7.1%), so that run's absolute gains (s2 +14.4%, g2
+20.2%) overstate the effect; the interleaved table above is the comparison
to use.

No memory change: no buffers were added; the device chain reuses
`draft_tokens`, `mtp_ids` and the sampler's argmax partials.

## Evidence, decision and limits

- Proposed reference wording, not applied (reference pages are owned by the
  orchestrator): in `docs/NATIVE_HYBRID_MODELS.md` and `docs/GENERATION.md`,
  greedy MTP requests no longer read complete F32 draft-head readouts; the
  device selects each greedy draft by first-max argmax and the host reads the
  proposal ids, rerunning the selection on the host when a selected logit is a
  zero or subnormal. Verify rows are still read as complete F32 readouts and
  validated when consumed.
- Observation outside scope: greedy decode with MTP off already uses the same
  device argmax (`sample_f32`). On a readout whose maximum is a zero or
  subnormal it can differ from the host rule. This is unreachable for normal
  model logits but is not covered by the existing finite-readout guard.
- Not done: merging the finish pass into the next round's first submission
  (out of scope; it would remove one commit and wait per round, roughly the
  0.3 ms gap between finish wall and GPU time).
- The trait method `draft_point_mass` likely overlaps the depth-policy
  workstream's planned point-mass method; the two should be unified.

## Follow-up history

None yet.

## Integration

2026-11-19. The adopted attempts were merged into one uncommitted working tree
(the `claude/sushi-stark-performance-598713` checkout, base `4d09b550`), in
this order: `bench.patch` (SHA-256 `95e3ca04...`, 20261119-LILYBENCH-01),
`gdn-replay-v1.patch` (`542bc1e5...`, 20261119-LILYROWS-02, without the later
projection variants), `host-roundtrip.patch` (`ff797c62...`,
20261119-LILYHOST-01) and `depth-policy.patch` (`683cb04f...`,
20261119-LILYDEPTH-01). Nothing is committed, installed or released; the final
quiet before/after timing is pending.

Merge resolutions:

- `MtpSampler::draft_point_mass` exists once. The kept implementation refuses
  a call after verification began and an id at or past a known vocabulary
  size, and, like `draft`, accepts a sampler whose vocabulary size is not yet
  known; the verify pass refuses ids outside the model vocabulary and
  `target` checks and sizes the readout before any proposal is used. In the
  worker, MTP requests always draw the first token through `sample_initial`,
  so the sampler is sized before the first round on both the device chain and
  lookup paths. Both authors' tests are kept (the host test's unsized case now
  expects acceptance) and `first_round_point_masses_before_the_sampler_is_sized`
  covers the first round with and without sizing.
- `MtpDecode::round`: a head plan of d > 0 drafts (fixed or adaptive) goes
  through `draft_host_encoding_verify`, so greedy requests use the device
  draft chain and every head round encodes its verify graph ahead; a lookup
  plan records point masses, calls `advance_head_host` before the last round
  and verifies with `verify_host_batched`; a scheduled zero-draft plan returns
  through `serial_round` before anything is encoded.
- `verify_host_encoded` takes the scratch mutably for the GDN shadow swap, and
  an encoded-ahead verify also checks that the GDN recurrent-state buffers it
  was encoded against are still current (the replay rollback swaps them).
- The depth planner's static cost table was measured before the host
  round-trip change and is unchanged; retuning is a separate decision.

Gates on the merged worker (Rust 1.98.1 release build, SHA-256
`d4182a6769e7c86c6865d0660ccc0d82d4d6442c774eb0f0bdbe83fd6c3041af`). All
timings are unscreened shared-host screening and are not quoted here.

| Gate | Result |
| --- | --- |
| Release build | Clean |
| stark-generate library tests | 153 passed, 5 ignored |
| stark-engine `hybrid` filter / `mtp_` filter | 285 passed, 73 ignored / 30 passed, 8 ignored |
| lily-core library tests | 98 passed, 58 ignored; plus 12 ignored physical-Metal byte tests run and passed (GDN verify, value-key verify, projection rows, chunk capture, attention verify, chained argmax and finite readout) |
| Benchmark script tests | 29 passed |
| `cargo fmt --check` (workspace crates) | Clean |
| `private_checkpoint_mtp_verify_and_rollback_match_serial`, real model | Passed, including the device-versus-host greedy section: verify 86, rollback 39 and host-serial 30 readouts with 0 unequal values; draft-head phase differs as allowed (0 top-1 mismatches) |
| Eight-arm output-hash gate (`runtime/agent-runs/integration-1.json`) | All eight output hashes equal the 20261119-SUSHI-02 baseline |
| New options, frozen corpus, two repeats (`integration-options-1.json`) | Greedy adaptive, lookup and adaptive plus lookup equal the baseline greedy hash; sampled adaptive, lookup and adaptive plus lookup one hash each across repeats. Sampled lookup had no lookup rounds on this input |
| New options, copy-heavy input, two repeats (`integration-options-2-copy.json`) | Greedy adaptive, lookup and adaptive plus lookup equal fixed two-draft greedy; sampled lookup (2 lookup rounds) and adaptive plus lookup (6) one hash each across repeats |

Not covered end to end: a scheduled zero-draft round in the middle of a
stream (only final-round zero-draft rounds occurred in these runs).
