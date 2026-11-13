# 20261113-EDGE0-10: worker integration and matched MTP measurement

## Identity and scope

| Field | Value |
| --- | --- |
| Registered / owner | 2026-11-13 / Codex, local non-commercial evaluation only |
| Baseline | `0418a3d3b69cc57c08c4e5814be0e545297d1832`, qualified private combined restoration and exact sampler |
| State / result | Serial-target worker lifecycle passed; fixed-work benchmark failed before measurement; superseded by attempt 11 |
| Governing plan | [Phase 5](../EDGE0_PLAN.md#phase-5-mtp), owner-approved native Qwen Metal graph |
| Predecessor | [20261113-EDGE0-09](2026-11-13-edge0-mtp-integration.md) |
| Promotion boundary | Explicit local worker opt-in only; no app/catalogue, distribution or installed-build promotion |

The owner requested worker integration and matched MTP on/off measurements.
This attempt connects the qualified one-to-three-draft complete-vocabulary exact
sampler to the ordinary JSONL worker. Verification remains ordinary single-token
target execution. Original stored weights and F32 arithmetic are unchanged.
The 512-token cap remains; prefix reuse stays unavailable because chunk-boundary
worker restoration, the wider-context finite concern and the 4,096-token gate
are unresolved. The artifact is text-only under the existing source inventory.

## Fixed correctness and lifecycle gates

- Readiness, request overrides and returned MTP acknowledgements must agree.
  Omitted MTP and explicit off select zero drafts. Explicit on selects the
  recommended depth; explicit depths one through three must produce real draft
  activity for a sufficiently long output. Four drafts are refused.
- The worker must check its existing request-ID cancellation lane throughout
  prompt ingestion, before and after head/target/restore/replay work, before
  delivering queued output and after final cleanup. GPU submissions complete
  before returning; asynchronous cancellation is observed at those boundaries.
  Cancellation, callback errors, channel failure and rollback failure invalidate
  the request's complete state and prevent a successful acknowledgement.
- Preserve ordinary EOS, extra stop IDs, text stops, streaming, output tails,
  progress and resource admission. Clean stop must restore the delivered boundary.
  Request B must recover on the same model after cancellation or rejection.
- Account for head/snapshot/replacement resources before dispatch. The internal
  512 MiB head lease remains bounded; worker planning may conservatively charge
  a further snapshot allowance. Source mappings are virtual-only for this attempt.
- Preserve existing model-free collector/controller/exact-sampler tests. Exercise
  the actual normal worker with explicit off, true and depths one through three,
  greedy and stochastic settings, short tails, stop/cancellation/recovery,
  invalid depths/context and a prefix-cache request. Keep all private content
  in RAM/pipes; publish aggregate results and hashes only.

## Artifact, build and resource boundary

The pinned artifact is `Edge0/Edge0-145B-q3k` at
`d7de835cfce7d9bb22a3d5ebb7d1e8565d4ecf74`, prepared under
`runtime/edge0-native-v2-evaluation/`. Its `native-source.json` SHA-256 remains
`8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e`.
Reuse the complete preparation hash receipt under the explicit large-payload
immutability assumption; do not label this a new 103 GB rehash.

Build the worker with locked dependencies and Rust 1.98.1. Seal source inputs,
executable, harness and exact configuration before launch and recheck afterward.
One owned worker/model process runs at a time on the 128 GiB Apple M5 Max host.
Require 64 GiB available before worker launch, sampled owned-tree RSS below
48 GiB, at most 180 seconds per request and 1,800 seconds per worker session.
No OS wired-memory setting or another session's process is changed. Resource
stops, source drift and failed cases are retained with their disposition.

## Matched performance recipe and default decision

Use the actual JSONL worker, one loaded model, no prefix reuse and a private
ordinary-text corpus bound by hash before launch. Use greedy sampling and seed 42
for comparable useful output. The fixed matrix is:

- Prompt lengths 32 and 128; output budget 16; context 512.
- MTP off versus depths one, two and three, each as a separate paired group.
- One warmup per route before each group; four measured pairs, with balanced
  ABBA leading-arm order selected deterministically by seed 42. No arm retries.
- Ten-second quiet admission before each group, then host screening of every
  measured arm: external CPU at most 150%, external model CPU at most 10%,
  thermal at most fair, no compiler, sampling failure/gap over two seconds,
  pageout, swapout, compression or decompression counter increase. Idle GPU
  utilization during quiet admission must be at most 5%.
- Both arms must emit the full 16-token budget with identical greedy output
  hashes and valid acknowledgement/counters. Early EOS or unequal useful work
  invalidates that group. Off must report zero MTP activity; on must report
  positive proposal activity. Retain all ineligible groups without ranking them.
- Report prefill/first-token, useful decode `(sampled_tokens - 1) / decode_seconds`,
  generation and IPC-inclusive wall time separately, with paired ratios and
  dispersion. Warmup and loading time are excluded from those matched summaries.
  Require off-route wall spread `(max - min) / median <= 0.15` per group for a
  default-selection claim.

MTP remains off by default unless a depth passes every correctness/screening gate
on both prompt lengths, has median paired useful-decode speedup at least 1.05,
median paired IPC-inclusive wall speedup at least 1.05, and median prefill time
no greater than 1.05 times off. Choose the passing depth with the strongest
minimum wall speedup across both prompts. Preserve the explicit request override.
If no depth qualifies, retain off and record the measured regression or lack of
qualification. Stochastic same-seed MTP on/off output identity is not a gate.

## Attempt history, results and disposition

The worker implementation preserves an explicit default of zero drafts and
advertises `serial-target-exact-v1` only for the opt-in virtual Metal path.
Explicit residency remains supported for ordinary target inference under Phase 4;
resident-mode worker MTP is unavailable in this attempt.

Prelaunch review corrected cancellation polls after a verify round and final
drain. The first production Clippy check reported three unnecessary lint
expectations and one iterator-loop style diagnostic; these were corrected and
the repeat passed. No inference arithmetic changed. Generator model-free tests
passed 150 with six ignored; final sealed-build regression remains to run.

The concrete normal-worker lifecycle recipe has 26 cases: omitted/false/zero/true
overrides; greedy and stochastic depths one through three; budgets one, two and
three; an extra stop ID; a text stop; streaming; rejected depth four, context 513
and prefix scope; memory planning off/on; request-ID cancellation during a
128-token prompt and after a delivered decode token, each followed by a same-model
greedy recovery with a stale cancellation ID queued; and cancellation-channel EOF.
Require nine greedy output-hash comparisons, two live request-ID cancellations,
two same-model recoveries and one EOF cancellation. The prompt cancellation frame
is split into three and five bytes. All three cancellation triggers must fire
during active work, with content-free error replies and no success acknowledgement.
The final worker must exit normally with code zero on stdin EOF; forced cleanup
is failure evidence. A later subprocess is never substituted for same-model recovery.

The matched benchmark is warm, same-process worker measurement. Loading, cold
first-request time and both route warmups are excluded. Concrete source/build
hashes, immutable receipts and the measured default decision will be recorded
before completion.

## Recorded results and successor

At worker source `6a349590` plus the capability-only diagnostic update `64d3ab12`,
the native unit suite passed 1,098 with 162 ignored; generator units passed 150
with six ignored; four head controls, production Metal Clippy and CPU feature
checking passed. The sealed release worker SHA-256 was
`55b86698e1c973d3dbc4e7b00a786687cc732199beffece4755731a996f92940`.
The 2,182-file source manifest was
`9f2028e9c00a3db1364256c38f4f7bbe5aa358e8d9b01271826a325c00b7696e`.

The normal worker passed all 26 registered cases and nine greedy output-hash
comparisons with both same-model recoveries. It exited on stdin EOF with code 0.
Peak sampled owned RSS was 840,204,288 bytes; available memory before launch was
96,742,309,888 bytes. The diagnostic independently passed its 15 cases and 41
complete target-row pairs, with 24 proposals and nine acceptances, against the
serial-prompt target policy. Its source-bound log SHA-256 was
`aae119b247731fdc40859479578556cf66ea9e192652b264048ed15ac5ba8813`.

Two limits prevent a full Phase 5 claim from these results. The cancellation
labelled decode fired after the initial prefill output, so it proves cancellation
after the first delivered token and recovery, without demonstrating a verified
tail output first. Also, off-route prompt ingestion used batches of eight while
this candidate's MTP prompt ingestion was serial. The oracle used a serial prompt;
cross-mode target-distribution identity is unverified for this revision.

The first off-route benchmark warmup failed `unequal_work` before emitting the
required full 16-token workload. No measured pair or speed result exists. The
partial group and normal process exit are preserved; the gate was not relaxed
retroactively and no default was promoted.

The successor [attempt `20261113-EDGE0-11`](2026-11-13-edge0-mtp-batched-worker.md)
aligns prompt arithmetic, strengthens the cancellation trigger, and separately
registers matching natural-stop requests. Default remains off.
