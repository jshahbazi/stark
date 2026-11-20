# 20261119-LILYROWS-02: Lily GDN verify replay rollback

## Identity and state

| Field | Value |
| --- | --- |
| Attempt ID / registered date / executed date / owner | `20261119-LILYROWS-02` / 2026-11-19 / 2026-11-19 / `gdn-replay` workstream of the Lily MTP speed project |
| Kind | Kernel, state-affecting rollback path, memory and performance |
| Attempt state | Completed (draft record, uncommitted worktree) |
| Result and acceptance criterion | Passed exactness: kernel byte tests, real-model verify/rollback control and the end-to-end output-hash gate on every arm. Performance target (GDN verify about 3.6/3.9/4.2 ms at 2/3/4 rows, slope at or under 0.3 ms per row) not met: the recurrence now meets it, the GDN input projection does not (see below) |
| Evidence scope | Operator/component (synthetic byte and timing), full-model numerical, lifecycle (early-stop drop path), end-to-end timing (screening, shared host) |
| Promotion decision | Replay rollback (`runtime/patches/gdn-replay-v1.patch`) adopted on the integration working tree; production promotion awaits the final quiet timing. The later projection-kernel variants (follow-up below) are rejected and not adopted |
| Disposition | Merged with 20261119-LILYHOST-01, 20261119-LILYDEPTH-01 and 20261119-LILYBENCH-01 into one uncommitted integration working tree on base `4d09b550`; not committed, not installed, final quiet timing pending. The follow-up projection variants are not on that working tree; their patch is archived as unadopted source in [`docs/archive/experiments/2026-11-19-lily-verify-rows/`](../archive/experiments/2026-11-19-lily-verify-rows/README.md). Integration gates are listed under [Integration](#integration) |
| Predecessor / related attempts / governing issue / PR | Baseline `20261119-SUSHI-02`; related: projection reuse (`2026-11-16-lily-projection-reuse.md`), small-row verifier (`docs/benchmarks/2026-11-15-lily-small-row-verifier.md`) |

## Hypothesis and change

- Problem: in `BatchMode::VerifyHost` the GDN recurrence kernel `gdn_verify_gated_exact_register` wrote a full F32 state after every row but the last (113,246,208 bytes per extra row over 36 layers) so a partial accept could copy one back. The baseline brief attributed the rising GDN verify slope (0.55, 1.24, 1.65 ms) to this.
- Diagnostic first step: the verify stage profile emitted only the parent `qwen.mtp.verify.gpu.gdn` because `block_batched` marked `Phase::Gdn` and `gdn_batched` (the verify/prefill GDN branch) had no sub-phase markers; only the decode branch (`gdn_decode_branch`) did. The exact verify branches of `gdn_batched` now mark input projection(+conv), conv, recurrence(+norm) and output projection, the same `Phase::Gdn*` children decode uses. Markers are no-ops unless stage profiling is enabled.
- Candidate (Sushi technique, reimplemented; no code copied): the verify recurrence reads the layer state and writes only the final state, into the other half of a per-layer double buffer (`SpecScratch::gdn_shadow`). The host swaps the two handles after the pass (`DecodeState::swap_gdn_shadow`, no data moved), so the shadow then holds the pre-verify state. The pass writes the rows' recurrence inputs (convolved q/k/v, `a`, `b`) straight into per-layer replay rows instead of prefill scratch. On a partial accept, `finish_host_verify` reruns the same compiled kernel (same pipeline, same arguments, prefix views) from the shadow over the committed rows into the current state; a full accept does nothing. The pre-verify state survives the first finish, so the early-stop `Drop` path can finish again with a smaller count. Row count is a kernel parameter (Rust limit `GDN_VERIFY_MAX_ROWS = 16`; tests run 1 to 9 rows). The ValueKey (official) verify kernel got the same in/out split and replay.
- Legacy GPU-sampled path (`verify`, `BatchMode::Verify`, not called anywhere in Stark) keeps per-row captures, now allocated lazily on its first use; the serial reference `verify_host` copies the state to the shadow and records replay rows so the shared `finish_host_verify` replays for it too.
- Controls set before running: byte identity to ordinary decode for outputs, final state, untouched pre-verify state, in-place form, and every replay prefix; real-model `private_checkpoint_mtp_verify_and_rollback_match_serial` with zero unequal values in verify, rollback and host-serial phases; identical output hashes to the frozen baseline on all eight arms.

## Artifact, source and environment

| Field | Baseline | Candidate |
| --- | --- | --- |
| Model family, exact checkpoint and upstream revision | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` at `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, installed copy, image tower loaded | same |
| Format/quantization and hashes | Lily Q4-XL, KeyValue GDN state layout; per-file hashes not re-recorded | same |
| Backend, vendor revision and local patch provenance | `vendor/lily-core` at base | Stark-local patch to `kernels/gdn.rs`, `kernels/metal/gdn.metal`, `qwen4exp/{model,spec,host_spec,request_memory}.rs` and GDN tests |
| Stark commit, dirty patch and executable hash | `4d09b55045f24eab81da0a9b5b024b9c02318459` plus the diagnostic markers only; worker SHA-256 `d42451b7cb067e91d72b2ff252e9ad421f959c5c873b0caa042c2c7561e2fd6d` (frozen baseline worker `5dfb90b6...` for hash gate) | same base plus this patch; worker SHA-256 `0b72de41f7076445949b00c41d2956b24bcb9f87954a979f5bb61d92f3d157eb`; checkpoint patch `runtime/patches/gdn-replay.patch` |
| Toolchains and build flags | Rust 1.98.1 `--release --locked`; nightly-2026-10-01 for engine tests | same |
| Precision | F32 recurrent state, BF16 activations, unchanged | unchanged |
| Kernel policy | `gdn_verify_gated_exact_register` with per-row `mid` captures | same kernel, state in/out split, no captures; replay reruns it |
| Hardware | Apple M5 Max, 128 GB | same |
| Memory | MTP scratch held `MAX_DRAFTS` F32 states per GDN layer (339,738,624 bytes) | one shadow state per GDN layer (113,246,208 bytes) plus replay rows (about 3.0 MB) and a 49 KB sink; saves about 226 MB of request scratch |

## Workload and controls

- Corpus and options: frozen 4,096-token public code corpus (input hash matched the baseline), seed 42, sampled temperature 1 / top-k 20 / top-p 0.95 for 256 tokens, greedy 512 tokens, drafts 0 to 3, image tower loaded, fresh requests.
- Paired order: baseline-equivalent worker then candidate, back to back inside one model-slot hold, two pairs for timing, one instrumented pair. Other agents shared the host; not quiet-screened. Timings are screening only.
- Image, cache and MTP-with-images qualification: not re-run; the change is confined to the text verify/rollback recurrence and leaves image requests (MTP off) untouched.

## Attempt history and failures

| Trial / source | What was attempted | Observation | Validity and disposition |
| --- | --- | --- | --- |
| `gdn-replay-1` (instrumented, base + markers) | Split GDN verify cost | Decode 1 row: projection+conv 1.78, recurrence 0.81, out projection 0.76 ms. Verify 2/3/4 rows: projection+conv 2.01/2.82/3.62, recurrence 1.79/1.89/2.40, out projection 0.92/1.14/1.55 | Valid attribution; the slope at 3 and 4 rows is mostly the input projection, not the captures. Recurrence has a large 1-to-2-row step |
| Synthetic timing, `gdn_verification_streaming_chain_timing` | Register verify (final state only) vs decode steps vs replay, 36 layers | Final-only verify 17.6/20.0/23.1/26.2/41.1 us per layer at 1/2/3/4/9 rows (about 0.11 ms per extra row over 36 layers); decode steps grow about 7 us per row per layer; replay of m-1 rows 16.6/19.6/23.6/37.8 | Component evidence for the recurrence |
| `stark_gdn_verify_gated_stream` | Device-memory variant through decode's own step body | 42 to 77 us per layer, about twice the register kernel | Rejected, removed |
| `gdn-replay-2` / `-3` | First candidate end-to-end runs | Hashes matched; host heavily contended (draft and s0 also slowed 25%) | Timing invalid, hash evidence kept |
| `gdn-replay-4p` (instrumented pair) | Base vs candidate | See table below | Valid screening |
| `gdn-replay-5t` (two uninstrumented pairs) | Base vs candidate, all arms | See table below; pair 0 noisier (g0 +8%, g3 +13%) | Pair 1 used for conclusions |

## Correctness and lifecycle

- `gdn_register_verification_matches_decode_bytes` (extended): three independent draws x two geometries x rows 1 to 9 x two gates x six styles (ordinary, zero q/k, saturated gates, signed zeros/extremes, subnormal-heavy state and inputs, cancellation); 648 cases, 3,240 replay prefixes, 9,217,732,608 bytes compared, reference and candidate SHA-256 `78d5256babf559b2799e35c899ac7258406803bfb17755c14d94241dff25d1b7`, all equal. Checks pre-verify state untouched, final state, in-place form, outputs and every replay prefix against ordinary decode.
- `fused_gdn_projection_rows_match_decode_and_all_rollback_prefixes` (extended): 128 cases, 448 conv rollback prefixes, 160 recurrence replay prefixes on the Lily geometry chained from the fused projection rows, SHA-256 `76324820eea39b0683c8e39d9b044bced9a572cc44cf0afd277b5b0211dbdd34` on both sides.
- `official_value_key_verification_rows_match_decode_and_captures` (ValueKey): three draws x rows 1/2/3/4/9, 57 replay prefixes, all equal. `gdn_verification_batch_matches_decode_bytes` (legacy control kernel) passes unchanged.
- Real model, `private_checkpoint_mtp_verify_and_rollback_match_serial` (reads `STARK_TEST_MODEL_PATH`): passed; verify phase 86 readouts, rollback phase 24 readouts, host-serial phase 30 readouts, all zero unequal values; it covers every accepted count at widths 1 to 4, the dense-to-sparse lengths, and the early-stop `Drop` path at 0 to 4 pulled tokens. The draft-head phase differs as before (head numerics are allowed to differ; not asserted).
- Model-free: lily-core unit suite 98 passed (includes new `mtp_scratch_charges_one_gdn_shadow_state_not_per_row_captures`); stark-generate prefix tests 25 passed, 1 ignored; GDN midpoint capture tests (`chunk_capture_tests`, synthetic Metal) 2 passed.
- Not run: `private_lily_midpoint_full_logits_and_mtp` and `private_lily_midpoint_lifecycle_transitions` need private fixtures (`STARK_LILY_MIDPOINT_FIXTURE`, `STARK_LILY_MIDPOINT_IMAGE`) whose location was not available to this workstream. Prefix snapshots copy the current state handle at call time and are not taken while a verify is pending, so the handle swap does not reach them, but this is code review, not a run.
- Buffer ownership: the swap exchanges handles between the request state and the model scratch; both live in the context's single residency set. Snapshots and restores copy data. A failed verify swaps nothing.

## Performance and memory

Instrumented pair (`gdn-replay-4p`, GPU ms per verify pass, rows = drafts + 1):

| Stage | g1 (2 rows) base / cand | g2 (3 rows) | g3 (4 rows) | s2 | s3 |
| --- | --- | --- | --- | --- | --- |
| GDN recurrence+norm | 1.850 / 1.651 | 2.078 / 1.711 | 2.386 / 1.846 | 2.132 / 1.879 | 2.398 / 1.942 |
| GDN total | 4.884 / 4.676 | 6.037 / 5.730 | 7.586 / 7.258 | 6.059 / 5.762 | 7.663 / 7.078 |
| Verify command | 20.53 / 20.29 | 25.90 / 25.25 | 31.31 / 31.11 | 25.80 / 25.12 | 31.49 / 30.10 |
| Finish GPU busy per round | 0.407 / 0.408 | 0.566 / 0.613 | 0.949 / 0.881 | 0.565 / 0.574 | 0.727 / 0.827 |

Uninstrumented pair 1 (`gdn-replay-5t-*-1`, tok/s): g1 82.44 to 84.76 (+2.8%), g2 90.33 to 93.03 (+3.0%), g3 88.83 to 90.88 (+2.3%), s1 +2.9%, s2 77.95 to 79.20 (+1.6%), s3 66.99 to 69.28 (+3.4%); g0 +0.5%, s0 0.0%. Verify per round fell 0.5 to 1.3 ms; finish per round changed by 0.00 to +0.11 ms (replay on partial accepts, largest on s3). The replay costs less than its verify saving in every arm, sampled arms included.

The target was not met for GDN as a whole: the recurrence now grows about 0.1 ms per row in the model (1.65, 1.71, 1.85 ms), but it still carries a fixed step of about 0.8 ms over decode's 0.83 ms at two rows in the full model (synthetic shows only 0.17 ms); and the fused Q4 input projection plus convolution (`stark_q4_gdn_conv_rows_m{2,3,4}`, `stark_gemv_q4_packed_rows_sum`) grows 0.75 to 1.0 ms per row at 3 and 4 rows, with the output projection adding 0.2 to 0.4 ms per row. Those projection kernels are shared row-reuse code outside this workstream's files and were not changed.

## Evidence, decision and limits

- Run files (aggregates and hashes only): `runtime/agent-runs/gdn-replay-1-*` (attribution), `gdn-replay-2.json` (hash gate, contended), `gdn-replay-3-*`, `gdn-replay-4p-{base,cand}-0*` (instrumented pair), `gdn-replay-5t-{base,cand}-{0,1}.json` (timing pairs). All candidate output hashes equal the frozen baseline on all eight arms.
- What else blocks nine verify rows on the GDN side: `gemv_q4_gdn_conv_rows_exact` is instantiated for 1 to 4 rows only and its per-row cost rises steeply; `MAX_DRAFTS`/`VERIFY_ROWS` size the replay rows and conv-input scratch; `prefix_rows`/`project_mat_decode_exact` row paths for the output projection; `conv_window_rollback` and the recurrence already take any row count. Per-row GDN state cost is now zero bytes, so nine rows add only about 0.4 MB of replay rows per verify.
- Next: a row-reuse rewrite of the GDN input projection (register pressure at M=3/4 is the likely cause) and an investigation of the in-model two-row recurrence step.

## Follow-up history

- 2026-11-19: draft record created by the workstream; orchestrator to integrate provenance and the quiet comparison.
- 2026-11-19, projection follow-up (same attempt ID): why rows 3 and 4 of the shared-weight Q4 projections cost more than row 2.
  - Synthetic component timing (`q4_row_reuse_projection_timing`, 36 distinct Lily-size weights) reproduces the real-model shape: GDN input projection+conv 1.66/1.78/2.21/2.96 ms and output projection 0.67/0.71/0.86/1.15 ms at 1/2/3/4 rows (quietest run; later runs on a busier host were 10 to 20% higher with the same shape). Real model: 1.78/2.01/2.82/3.62 and 0.76/0.92/1.14/1.55.
  - Ruled out: a second pass or lane change at m > 2 (every instance uses the same grid, 16 lanes per channel, one pass), a fallback path (m = 2..4 all take `stark_q4_gdn_conv_rows_m*` / `stark_projection_reuse_q4_bf16_m*`), and a thread-limit drop from registers (every pipeline reports 1,024 maximum threads).
  - Cause: per-row work that every output channel repeats. Each channel pair re-reads every input row from the device cache and re-does the BF16 widening and the weight-independent input sum of `dot_word_q4` for every word. One row costs about 1.66 ms of weight streaming (near DRAM bandwidth), which hides the per-row work at two rows. From three rows that work exceeds the weight stream and the kernel becomes compute and cache-load bound.
  - Restructurings, all byte-identical in synthetic tests (288 cases: 6 draws, ordinary, subnormal-heavy, cancellation and extreme inputs, offset views, rows 2 to 4, Lily and small geometries, against ordinary decode or the byte-tested fused kernel): (1) staged F32 rows and word sums from a pre-pass: slower (3.33/4.42 against 2.84/3.79 ms at 3/4 rows), rejected; (2) rows parked in threadgroup memory per 16 channels: 2.64/3.27 against 2.64/3.43, marginal, not pursued; (3) rows and word sums parked: 2.36/2.97 against 2.64/3.43, and 2.38/2.88 against 2.64/3.55 in the real model at 3/4 rows, but the real-model control `private_checkpoint_mtp_verify_and_rollback_match_serial` failed (verify phase 3,476,470 of 21,355,520 values unequal) and every MTP arm's output hash changed (`gdn-replay-8t-cand-0.json`). Rejected. Three restructurings done, so the work stopped there. The synthetic tests passed while the real model failed, which matches the fast-math scheduling fragility the HC workstream found. The variants remain in the source, test-only and switched off; the shipped default is the original kernel.
  - One-to-two-row recurrence step: fully unrolling the register kernel's state load and store loops was byte-identical in synthetic tests but slower in the real model (recurrence 2.48/2.84/3.02 against 1.98/2.16 ms). Reverted, cause not established. Quieter pair `gdn-replay-7p-base-0`: replay build recurrence 1.45/1.84/2.00 ms at 2/3/4 rows against 0.83 ms for the serial step.
  - Integration note: the variants above exist only in the workstream worktree and in the archived `gdn-replay-v2-projections.patch`; the integration working tree carries `gdn-replay-v1.patch` (replay rollback only) and none of these kernels.
  - Final state: the shipped behavior is the replay build plus test-only kernels. The real-model control passed again (verify, rollback and host-serial phases zero unequal). The end-to-end hash gate was not rerun on the final build; the replay build (`gdn-replay-8t-base-0`, same shipped kernels) matched all eight baseline hashes.

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
